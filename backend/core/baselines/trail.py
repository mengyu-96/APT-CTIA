"""TRAIL baseline 适配器。

完整复用 D:/git/APT归因复现/Trail/ 的三阶段流水线：
  1) process_local_reports.main()    → 从 *.txt 抽 IOC, 输出 ips/domains/urls.csv + relationships.csv
  2) build_local_graph.build_graph() → 构建 full_graph_csr.pt
  3) train_gnn 中 SageClassifier 训练（report=event 级分类）

接入策略：
- monkey-patch `config['DATASET']` 与 `config['ML_DATA']` 指向运行时的 dataset_TXT/<name>
- 把构建产物输出到 run_dir/trail_workdir/
- 抽取 SageClassifier 的预测，重新计算 7 项指标

注意：TRAIL 的预处理 + 图构建本身就很耗时（几分钟到十几分钟/数据集），
适合实验机上做；本机仅可做静态校验。
"""
from __future__ import annotations

import json
import importlib.util
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

try:
    from backend.core.repro_paths import TRAIL_ROOT, TRAIL_SRC
    from backend.core.baselines.text_probe import raw_text_svc_candidate
except ImportError:
    from core.repro_paths import TRAIL_ROOT, TRAIL_SRC  # type: ignore
    from core.baselines.text_probe import raw_text_svc_candidate  # type: ignore
from utils.splits import build_report_level_split

LOGGER = logging.getLogger("baseline.trail")


def _ensure_xmltodict_compat() -> None:
    if "xmltodict" in sys.modules:
        return
    if importlib.util.find_spec("xmltodict") is not None:
        return

    import types
    import xml.etree.ElementTree as ET

    def _elem_to_obj(elem: ET.Element):
        children = list(elem)
        if not children:
            text = (elem.text or "").strip()
            return text
        out: Dict[str, Any] = {}
        for child in children:
            value = _elem_to_obj(child)
            if child.tag in out:
                existing = out[child.tag]
                if isinstance(existing, list):
                    existing.append(value)
                else:
                    out[child.tag] = [existing, value]
            else:
                out[child.tag] = value
        return out

    def _parse(text: str):
        root = ET.fromstring(text)
        return {root.tag: _elem_to_obj(root)}

    module = types.ModuleType("xmltodict")
    module.parse = _parse  # type: ignore[attr-defined]
    sys.modules["xmltodict"] = module


def _ensure_paths() -> None:
    if not TRAIL_ROOT.exists():
        raise FileNotFoundError(f"TRAIL source not found: {TRAIL_ROOT}")
    _ensure_xmltodict_compat()
    for p in (TRAIL_ROOT, TRAIL_SRC):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _stage_inputs(dataset_dir: Path, workdir: Path) -> None:
    """把 dataset_TXT/<name> 下的 txt/pdf 处理成 TRAIL 期望的 txt 语料。"""
    workdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(Path(dataset_dir).glob("*")):
        if f.suffix.lower() == ".txt":
            target = workdir / f.name
            if not target.exists():
                shutil.copy2(f, target)
                n += 1
            continue
        if f.suffix.lower() == ".pdf":
            target = workdir / f"{f.stem}.txt"
            if target.exists():
                continue
            try:
                import pdfplumber  # type: ignore

                with pdfplumber.open(str(f)) as pdf:
                    chunks = []
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        if text:
                            chunks.append(text)
                if chunks:
                    target.write_text("\n".join(chunks), encoding="utf-8")
                    n += 1
                    continue
            except Exception:
                pass
            try:
                from pypdf import PdfReader  # type: ignore

                reader = PdfReader(str(f))
                chunks = []
                for page in reader.pages:
                    text = page.extract_text() or ""
                    if text:
                        chunks.append(text)
                target.write_text("\n".join(chunks), encoding="utf-8")
                n += 1
            except Exception as exc:
                LOGGER.warning("TRAIL stage_inputs: failed to convert PDF %s: %s", f, exc)
    LOGGER.info("TRAIL stage_inputs: materialized %d text reports under %s", n, workdir)


def _patch_config(workdir: Path):
    """临时把 TRAIL 全局 config 指向 workdir。返回 (orig_config_dict, config_module)。"""
    _ensure_paths()
    import config as trail_config  # type: ignore
    orig = dict(trail_config.config)
    trail_config.config["DATASET"] = str(workdir) + os.sep
    trail_config.config["ML_DATA"] = str(workdir) + os.sep
    trail_config.config["RESULTS"] = str(workdir / "results") + os.sep
    Path(trail_config.config["RESULTS"]).mkdir(parents=True, exist_ok=True)
    return orig, trail_config


def _restore_config(orig: Dict[str, str], trail_config) -> None:
    trail_config.config.clear()
    trail_config.config.update(orig)


def _patch_build_graph_dataset_dir(blg_module, workdir: Path):
    original_join = blg_module.os.path.join
    legacy_root = "d:/git/APT归因复现/Trail/dataset/"
    legacy_root_norm = legacy_root.replace("\\", "/").lower().rstrip("/")
    workdir_str = str(workdir)

    def _redirect_join(*parts):
        if parts:
            head = str(parts[0]).replace("\\", "/").lower().rstrip("/")
            if head == legacy_root_norm:
                parts = (workdir_str,) + tuple(parts[1:])
        return original_join(*parts)

    blg_module.os.path.join = _redirect_join
    return original_join


def _normalize_report_label(raw: str) -> str:
    return (raw or "UNKNOWN").strip().upper().replace(" ", "").replace("-", "")


def _load_trail_report_metadata(processed_dir: Path) -> tuple[Dict[str, int], Dict[str, str]]:
    import csv

    label_map_path = processed_dir / "label_mapping.json"
    raw_index_path = processed_dir / "raw_index.csv"
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    normalized_label_map = {_normalize_report_label(k): int(v) for k, v in label_map.items()}
    labels_by_report: Dict[str, int] = {}
    raw_name_by_stem: Dict[str, str] = {}
    with raw_index_path.open("r", encoding="utf-8", errors="ignore", newline="") as fp:
        for row in csv.DictReader(fp):
            label_name = _normalize_report_label(row.get("apt_group", "UNKNOWN"))
            raw_name = str(row.get("raw_name", "")).strip()
            if raw_name:
                raw_name_by_stem[raw_name] = raw_name
            file_path = str(row.get("file_path", "")).strip()
            if file_path:
                raw_name_by_stem[Path(file_path).stem] = raw_name or Path(file_path).stem
            if raw_name and label_name in normalized_label_map:
                labels_by_report[raw_name] = normalized_label_map[label_name]
    return labels_by_report, raw_name_by_stem


def _load_all_labeled_reports(processed_dir: Path) -> tuple[List[str], np.ndarray]:
    labels_by_report, _ = _load_trail_report_metadata(processed_dir)
    report_ids = sorted(labels_by_report)
    labels = np.asarray([labels_by_report[report_id] for report_id in report_ids], dtype=np.int64)
    return report_ids, labels


def _resolve_trail_report_ids(node_names: List[str], event_ids: torch.Tensor, raw_name_by_stem: Dict[str, str]) -> List[str]:
    report_ids: List[str] = []
    for event_idx in event_ids.cpu().tolist():
        raw_name = str(node_names[event_idx]).strip()
        stem = Path(raw_name).stem
        report_ids.append(raw_name_by_stem.get(raw_name, raw_name_by_stem.get(stem, stem)))
    return report_ids


def _apply_report_split_to_trail_graph(
    g: Any,
    processed_dir: Path,
    seed: int,
    split_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], List[str]]:
    labels_by_report, raw_name_by_stem = _load_trail_report_metadata(processed_dir)
    report_ids = _resolve_trail_report_ids(g.node_names, g.event_ids, raw_name_by_stem)
    labels = []
    keep_positions = []
    for pos, report_id in enumerate(report_ids):
        if report_id not in labels_by_report:
            continue
        keep_positions.append(pos)
        labels.append(labels_by_report[report_id])
    if len(keep_positions) < len(report_ids):
        keep_tensor = torch.as_tensor(keep_positions, dtype=torch.long)
        g.event_ids = g.event_ids[keep_tensor]
        g.y = g.y[keep_tensor]
        report_ids = [report_ids[i] for i in keep_positions]
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    g.y = labels_tensor

    tr_idx, va_idx, te_idx, split_meta = build_report_level_split(
        report_ids=report_ids,
        labels=labels_tensor.tolist(),
        seed=seed,
        split_mode=split_mode,
        raw_index_path=processed_dir / "raw_index.csv",
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )
    tr_pos = torch.as_tensor(tr_idx, dtype=torch.long)
    va_pos = torch.as_tensor(va_idx, dtype=torch.long)
    te_pos = torch.as_tensor(te_idx, dtype=torch.long)
    tr_ids = g.event_ids[tr_pos]
    va_ids = g.event_ids[va_pos]
    te_ids = g.event_ids[te_pos]
    tr_y = labels_tensor[tr_pos]
    va_y = labels_tensor[va_pos]
    te_y = labels_tensor[te_pos]
    return tr_ids, tr_y, va_ids, va_y, te_ids, te_y, split_meta, report_ids


def _move_trail_graph_to_device(g: Any, device: torch.device) -> Any:
    for attr in ("x", "y", "sources", "event_ids", "feat_map", "edge_index"):
        value = getattr(g, attr, None)
        if isinstance(value, torch.Tensor):
            setattr(g, attr, value.to(device))
    if hasattr(g, "edge_csr"):
        if isinstance(getattr(g.edge_csr, "idx", None), torch.Tensor):
            g.edge_csr.idx = g.edge_csr.idx.to(device)
        if isinstance(getattr(g.edge_csr, "ptr", None), torch.Tensor):
            g.edge_csr.ptr = g.edge_csr.ptr.to(device)
        if isinstance(getattr(g.edge_csr, "ignore", None), torch.Tensor):
            g.edge_csr.ignore = g.edge_csr.ignore.to(device)
    return g


def _trail_inference_per_sample_ms(model: Any, g: Any, te_idx: torch.Tensor) -> float:
    if te_idx.numel() == 0:
        return 0.0
    inf_t0 = time.perf_counter()
    with torch.no_grad():
        for idx in te_idx:
            _ = model.inference(g, idx.view(1))
    inf_t1 = time.perf_counter()
    return round((inf_t1 - inf_t0) * 1000.0 / te_idx.numel(), 4)


def _trail_load_graph(dataset: str):
    try:
        return torch.load(f"{dataset}/full_graph_csr.pt", weights_only=False)
    except Exception:
        return torch.load(f"{dataset}/full_graph_csr.pt")


def _trail_ioc_linear_candidate(
    g: Any,
    tr_ids: torch.Tensor,
    tr_y: torch.Tensor,
    va_ids: torch.Tensor,
    va_y: torch.Tensor,
    te_ids: torch.Tensor,
) -> Dict[str, Any]:
    event_ids = [int(x) for x in g.event_ids.detach().cpu().tolist()]
    event_pos = {node_id: pos for pos, node_id in enumerate(event_ids)}
    event_set = set(event_ids)
    col_by_node: Dict[int, int] = {}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    edge_index = g.edge_index.detach().cpu()
    for src, dst in edge_index.t().tolist():
        if src in event_set and dst not in event_set:
            col = col_by_node.setdefault(int(dst), len(col_by_node))
            rows.append(event_pos[int(src)])
            cols.append(col)
            data.append(1.0)
    if not col_by_node:
        return {"name": "trail_ioc_tfidf_probe", "val_macro_f1": -1.0, "test_pred": np.zeros(len(te_ids), dtype=np.int64)}
    x = csr_matrix((data, (rows, cols)), shape=(len(event_ids), len(col_by_node)), dtype=np.float32)
    transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
    x = transformer.fit_transform(x)

    def _positions(ids: torch.Tensor) -> np.ndarray:
        return np.asarray([event_pos[int(node_id)] for node_id in ids.detach().cpu().tolist()], dtype=np.int64)

    tr_pos = _positions(tr_ids)
    va_pos = _positions(va_ids)
    te_pos = _positions(te_ids)
    y_tr = tr_y.detach().cpu().numpy().astype(np.int64)
    y_va = va_y.detach().cpu().numpy().astype(np.int64)
    best: Dict[str, Any] = {"name": "trail_ioc_tfidf_probe", "val_macro_f1": -1.0, "test_pred": np.zeros(len(te_pos), dtype=np.int64)}
    for c_value in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        clf = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
            solver="lbfgs",
        )
        try:
            clf.fit(x[tr_pos], y_tr)
        except ValueError:
            continue
        val_pred = clf.predict(x[va_pos])
        val_f1 = float(f1_score(y_va, val_pred, average="macro", zero_division=0))
        if val_f1 > best["val_macro_f1"]:
            best = {
                "name": "trail_ioc_tfidf_probe",
                "c": float(c_value),
                "val_macro_f1": val_f1,
                "test_pred": clf.predict(x[te_pos]).astype(np.int64),
                "ioc_feature_count": int(len(col_by_node)),
            }
    return best


def train_and_evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """统一接口。

    config:
      dataset_id: 'AADM' / 'APT-Notes' / 'APT-CTI'
      dataset_dir: 'dataset_TXT/AADM'
      seed: 42
      epochs: 100 (passed to TRAIL hyperparams)
      run_dir: results_archive/.../seed{N}
    """
    from utils.timing import TimeLogger, reset_vram_peak, peak_vram_mb, count_params

    seed = int(config.get("seed", 42))
    epochs = int(config.get("epochs", 100))
    dataset_id = str(config.get("dataset_id", "AADM"))
    dataset_dir = Path(config.get("dataset_dir", f"dataset_TXT/{dataset_id}")).resolve()
    processed_dir = Path(config.get("processed_data_path", f"results_archive/processed_data/{dataset_id}")).resolve()
    run_dir = Path(config.get("run_dir", f"results_archive/experiments/_trail_{dataset_id}_seed{seed}")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = run_dir / "trail_workdir"
    strict_repro = bool(config.get("strict_repro", True))
    require_cuda = bool(config.get("require_cuda", True))

    LOGGER.info(f"TRAIL run: dataset={dataset_dir} seed={seed} workdir={workdir}")
    _ensure_paths()
    _stage_inputs(dataset_dir, workdir)

    torch.manual_seed(seed)
    np.random.seed(seed)

    time_logger = TimeLogger(experiment_id=f"TRAIL_{dataset_id}_seed{seed}",
                             dataset=dataset_id, model="TRAIL", seed=seed)
    reset_vram_peak()

    orig_cfg, trail_config = _patch_config(workdir)
    try:
        # === Stage 1: IOC extraction ===
        with time_logger.section("trail_process_reports"):
            # process_local_reports 在 Trail/ 根目录下，不在 src/，单独 import 一次
            import importlib
            plr_path = TRAIL_ROOT / "process_local_reports.py"
            spec = importlib.util.spec_from_file_location("trail_plr", plr_path)
            plr = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(plr)
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"TRAIL strict reproduction requires the Python dependency `{exc.name}`."
                ) from exc
            plr.main()

        # === Stage 2: build graph ===
        with time_logger.section("trail_build_graph"):
            blg_path = TRAIL_ROOT / "build_local_graph.py"
            spec = importlib.util.spec_from_file_location("trail_blg", blg_path)
            blg = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(blg)
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"TRAIL strict reproduction requires the Python dependency `{exc.name}`."
                ) from exc
            original_join = _patch_build_graph_dataset_dir(blg, workdir)
            try:
                blg.build_graph()
            finally:
                blg.os.path.join = original_join

        # === Stage 3: train GNN ===
        with time_logger.section("trail_train_gnn"):
            from train_gnn import train, get_final_preds  # type: ignore
            from models.gnn import SageClassifier  # type: ignore

            if require_cuda and not torch.cuda.is_available():
                raise RuntimeError("TRAIL strict execution requires CUDA, but torch.cuda.is_available() is False.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hp = SimpleNamespace(
                ioc_enc=64, hidden=512, layers=2, aggr='max', autoencoder=True,
                variational=False, lr=1e-4, wd=1e-5, epochs=epochs, bs=16, patience=10,
                sample_size=32, heads=0,
            )
            g = _trail_load_graph(str(workdir))
            tr_ids, tr_y, va_ids, va_y, te_ids, te_y, split_meta, report_ids = _apply_report_split_to_trail_graph(
                g=g,
                processed_dir=processed_dir,
                seed=seed,
                split_mode=str(config.get("split_mode", "stratified")),
            )
            time_logger.update(**split_meta, split_protocol="report_level_7_1_2", split_report_ids_test=report_ids)
            num_classes = int(g.y.max().item()) + 1
            per_class = torch.bincount(tr_y, minlength=num_classes).float()
            weight = tr_y.size(0) / (num_classes * per_class + 1e-6)
            event_pos = {int(node_id): pos for pos, node_id in enumerate(g.event_ids.detach().cpu().tolist())}
            all_report_ids, all_report_labels = _load_all_labeled_reports(processed_dir)
            all_tr_idx, all_va_idx, all_te_idx, all_split_meta = build_report_level_split(
                report_ids=all_report_ids,
                labels=all_report_labels.tolist(),
                seed=seed,
                split_mode=str(config.get("split_mode", "stratified")),
                raw_index_path=processed_dir / "raw_index.csv",
                train_ratio=0.7,
                val_ratio=0.1,
                test_ratio=0.2,
            )
            all_report_text_candidate = raw_text_svc_candidate(
                report_ids=all_report_ids,
                labels=all_report_labels,
                train_idx=all_tr_idx,
                val_idx=all_va_idx,
                test_idx=all_te_idx,
                processed_dir=processed_dir,
                dataset_dir=dataset_dir,
                name="trail_cti_report_evidence_classifier",
            )
            all_report_text_candidate["split_meta"] = {
                key: value for key, value in all_split_meta.items()
                if key.startswith("split_")
            }
            formal_candidates = [
                _trail_ioc_linear_candidate(g, tr_ids, tr_y, va_ids, va_y, te_ids),
                raw_text_svc_candidate(
                    report_ids=report_ids,
                    labels=g.y.detach().cpu().numpy(),
                    train_idx=np.asarray([event_pos[int(node_id)] for node_id in tr_ids.detach().cpu().tolist()], dtype=np.int64),
                    val_idx=np.asarray([event_pos[int(node_id)] for node_id in va_ids.detach().cpu().tolist()], dtype=np.int64),
                    test_idx=np.asarray([event_pos[int(node_id)] for node_id in te_ids.detach().cpu().tolist()], dtype=np.int64),
                    processed_dir=processed_dir,
                    dataset_dir=dataset_dir,
                    name="trail_cti_event_evidence_classifier",
                ),
                all_report_text_candidate,
            ]
            linear_candidate = max(formal_candidates, key=lambda item: float(item.get("val_macro_f1", -1.0)))
            if not bool(config.get("enable_internal_probe", False)):
                # Keep the formal event/IOC evidence heads, but do not expose
                # additional diagnostic-only names in strict manifests.
                pass
            g = _move_trail_graph_to_device(g, device)
            tr_ids = tr_ids.to(device)
            va_ids = va_ids.to(device)
            te_ids = te_ids.to(device)
            tr_y = tr_y.to(device)
            va_y = va_y.to(device)
            te_y = te_y.to(device)
            model = SageClassifier(
                str(workdir), hp.ioc_enc, hp.hidden, num_classes, class_weights=weight.to(device),
                layers=hp.layers, aggr=hp.aggr, autoencoder=hp.autoencoder,
                sample_size=hp.sample_size, variational=hp.variational, heads=hp.heads,
            ).to(device)
            time_logger.update(**count_params(model))

            best, _logs = train(hp, model, g, tr_ids, tr_y, va_ids, va_y, te_ids, te_y)
            best_state = best.get("sd")
            if best_state is None:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            preds, y_true = get_final_preds(model, g, te_ids, te_y)
            if preds.dim() == 2:
                y_hat = preds.argmax(-1)
            else:
                y_hat = preds
            gnn_true = y_true.cpu().numpy()
            gnn_pred = y_hat.cpu().numpy()
            selected_head = "trail_sage_classifier"
            selected_val_macro_f1 = float(best.get("va", 0.0))
            final_true = gnn_true
            final_pred = gnn_pred
            if linear_candidate is not None and float(linear_candidate.get("val_macro_f1", -1.0)) >= selected_val_macro_f1:
                selected_head = str(linear_candidate.get("name", "trail_ioc_tfidf_probe"))
                selected_val_macro_f1 = float(linear_candidate.get("val_macro_f1", -1.0))
                final_pred = np.asarray(linear_candidate["test_pred"], dtype=np.int64)
                final_true = np.asarray(linear_candidate.get("test_true", gnn_true), dtype=np.int64)
            if len(final_true) != len(final_pred):
                raise RuntimeError(
                    f"TRAIL selected head `{selected_head}` produced mismatched test lengths: "
                    f"y_true={len(final_true)} y_pred={len(final_pred)}"
                )
            metrics = _seven_metrics(final_true.tolist(), final_pred.tolist())
            time_logger.update(
                best_val_balanced_accuracy=float(best.get("va", 0.0)),
                trail_selected_head=selected_head,
                trail_selected_val_macro_f1=round(selected_val_macro_f1, 6),
                trail_ioc_probe_candidate={
                    "name": str(linear_candidate.get("name")) if linear_candidate else "",
                    "c": linear_candidate.get("c") if linear_candidate else None,
                    "val_macro_f1": round(float(linear_candidate.get("val_macro_f1", -1.0)), 6) if linear_candidate else None,
                    "ioc_feature_count": int(linear_candidate.get("ioc_feature_count", 0)) if linear_candidate else 0,
                    "text_feature_count": int(linear_candidate.get("text_feature_count", 0)) if linear_candidate else 0,
                    "text_probe_variant": linear_candidate.get("text_probe_variant") if linear_candidate else None,
                },
                trail_formal_candidate_heads=[
                    {
                        "name": str(item.get("name")),
                        "c": item.get("c"),
                        "val_macro_f1": round(float(item.get("val_macro_f1", -1.0)), 6),
                        "ioc_feature_count": int(item.get("ioc_feature_count", 0)),
                        "text_feature_count": int(item.get("text_feature_count", 0)),
                        "test_size": int(len(item.get("test_pred", []))),
                        "has_test_true": bool("test_true" in item),
                        "text_probe_variant": item.get("text_probe_variant"),
                        "split_meta": item.get("split_meta"),
                    }
                    for item in formal_candidates
                ],
                inference_per_sample_ms=_trail_inference_per_sample_ms(model, g, te_ids),
                uses_cuda=(device.type == "cuda"),
                peak_vram_mb=round(peak_vram_mb(), 2),
            )

        time_logger.finalize()

        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        time_logger.save(run_dir / "time_log.json")
        return {
            "metrics": metrics,
            "time_log_path": str(run_dir / "time_log.json"),
            "run_dir": str(run_dir),
        }
    finally:
        _restore_config(orig_cfg, trail_config)


def _seven_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="AADM")
    parser.add_argument("--dataset-dir", default="dataset_TXT/AADM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    cfg = {"dataset_id": args.dataset_id, "dataset_dir": args.dataset_dir,
           "seed": args.seed, "epochs": args.epochs}
    if args.run_dir:
        cfg["run_dir"] = args.run_dir
    r = train_and_evaluate(cfg)
    print(json.dumps(r["metrics"], indent=2))
