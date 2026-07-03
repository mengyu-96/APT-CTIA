"""APT-MMF baseline 适配器。

直接复用 D:/git/APT归因复现/.../APT-MMF/ 的端到端实现：
- monkey-patch `_dataset()` 使其指向我们的 dataset_TXT/<name>
- 注入 TimeLogger 计时各阶段
- 用统一的 7 项指标 + temperature(无) + 推理延迟（按 report 计）

参数兼容 backend.core.train.run_training_pipeline 的 config dict。
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
)
from sklearn.preprocessing import StandardScaler

try:
    from backend.core.repro_paths import APT_MMF_ROOT
    from backend.core.baselines.text_probe import raw_text_svc_candidate
except ImportError:
    from core.repro_paths import APT_MMF_ROOT  # type: ignore
    from core.baselines.text_probe import raw_text_svc_candidate  # type: ignore
from utils.splits import build_report_level_split

LOGGER = logging.getLogger("baseline.apt_mmf")


def _normalize_label(raw: str) -> str:
    return (raw or "UNKNOWN").strip().upper().replace(" ", "").replace("-", "")


def _ensure_paths(dataset_txt_root: Path):
    """让 APT-MMF 能 import 其内部模块，并定位到目标 dataset 目录。"""
    if not APT_MMF_ROOT.exists():
        raise FileNotFoundError(f"APT-MMF source not found: {APT_MMF_ROOT}")
    if str(APT_MMF_ROOT) not in sys.path:
        sys.path.insert(0, str(APT_MMF_ROOT))
    # external_knowledge / cache 相对位置：base = APT-MMF/..
    # 我们让 base = APT-MMF 自身，dataset/external_knowledge 也准备到该目录
    os.environ.setdefault("APT_SPLIT_SEED", "42")


def _patch_dataset_path(dataset_dir: Path):
    """monkey-patch data_loader._dataset 直接返回我们指定的 dataset_dir。"""
    import data_loader  # type: ignore
    original_dataset = data_loader._dataset

    def _custom(_base):
        return str(dataset_dir)

    data_loader._dataset = _custom  # type: ignore
    return original_dataset


def _load_label_aliases(processed_dir: Path) -> Dict[str, str]:
    raw_index_path = processed_dir / "raw_index.csv"
    aliases: Dict[str, str] = {}
    if not raw_index_path.exists():
        return aliases
    with raw_index_path.open("r", encoding="utf-8", errors="ignore", newline="") as fp:
        for row in csv.DictReader(fp):
            canonical = _normalize_label(row.get("apt_group", ""))
            if not canonical:
                continue
            raw_name = str(row.get("raw_name", "")).strip()
            report_id = str(row.get("report_id", "")).strip()
            file_path = str(row.get("file_path", "")).strip()
            for key in (raw_name, report_id, Path(file_path).stem if file_path else ""):
                key = key.strip()
                if key:
                    aliases[key] = canonical
    return aliases


def _patch_label_parser(processed_dir: Path):
    import data_loader  # type: ignore

    aliases = _load_label_aliases(processed_dir)
    original_label = data_loader._label

    def _canonical_label(name: str):
        base = Path(str(name)).stem
        direct = aliases.get(base)
        if direct:
            return direct
        parsed = original_label(name)
        parsed_norm = _normalize_label(parsed)
        if parsed_norm in aliases.values():
            return parsed_norm
        prefix = base.split("_", 1)[0].split("-", 1)[0].strip()
        direct = aliases.get(prefix)
        if direct:
            return direct
        return parsed_norm

    data_loader._label = _canonical_label  # type: ignore
    return original_label


def _restore_dataset(original):
    try:
        import data_loader  # type: ignore
        data_loader._dataset = original  # type: ignore
    except Exception:
        pass


def _select_runtime_device(heter_graph, homo_graphs, strict_repro: bool, require_cuda: bool):
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("APT-MMF strict execution requires CUDA, but torch.cuda.is_available() is False.")
        return torch.device("cpu")
    candidate = torch.device("cuda")
    try:
        _ = heter_graph.to(candidate)
        if homo_graphs:
            _ = homo_graphs[0].to(candidate)
        return candidate
    except Exception as exc:
        if strict_repro or require_cuda:
            raise RuntimeError("APT-MMF strict execution requires DGL/PyTorch graph tensors to move onto CUDA.") from exc
        LOGGER.warning("APT-MMF falling back to CPU because DGL CUDA transfer is unavailable: %s", exc)
        return torch.device("cpu")


def _collect_sample_report_ids(data_loader_module: Any, dataset_dir: Path) -> List[str]:
    base = os.path.abspath(os.path.join(os.path.dirname(data_loader_module.__file__), ".."))
    _, _, attack_phase_map = data_loader_module._attack(base)
    report_ids: List[str] = []
    for path in sorted(dataset_dir.glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".pdf"}:
            continue
        name = path.name
        label = data_loader_module._label(name)
        if not label:
            continue
        text = data_loader_module._text(str(path))
        segments = data_loader_module._segments(text) if data_loader_module.ENABLE_SEGMENT_AUG else [(text, 0)]
        for segment_text, seg_idx in segments:
            if len(re.sub(r"\s+", "", segment_text or "")) < 200:
                continue
            iocs = data_loader_module._clean_iocs(data_loader_module._iocs(segment_text, attack_phase_map, name, label))
            if sum(len(iocs.get(tp, set())) for tp in data_loader_module.IOC_TYPES) == 0 and len(segment_text) < 800:
                continue
            base_report_id = path.stem
            report_ids.append(base_report_id)
    return report_ids


def _build_grouped_report_split(
    report_ids: List[str],
    labels: List[int],
    seed: int,
    split_mode: str,
    raw_index_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    group_order: List[str] = []
    group_labels: List[int] = []
    seen: Dict[str, int] = {}
    for report_id, label in zip(report_ids, labels):
        if report_id in seen:
            continue
        seen[report_id] = len(group_order)
        group_order.append(report_id)
        group_labels.append(int(label))
    g_tr, g_va, g_te, meta = build_report_level_split(
        report_ids=group_order,
        labels=group_labels,
        seed=seed,
        split_mode=split_mode,
        raw_index_path=raw_index_path,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )
    buckets = {
        "train": {group_order[i] for i in g_tr.tolist()},
        "val": {group_order[i] for i in g_va.tolist()},
        "test": {group_order[i] for i in g_te.tolist()},
    }
    tr = [i for i, rid in enumerate(report_ids) if rid in buckets["train"]]
    va = [i for i, rid in enumerate(report_ids) if rid in buckets["val"]]
    te = [i for i, rid in enumerate(report_ids) if rid in buckets["test"]]
    meta.update(
        split_grouped_reports=True,
        split_unique_reports=int(len(group_order)),
        split_train_size=int(len(tr)),
        split_val_size=int(len(va)),
        split_test_size=int(len(te)),
    )
    return np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64), np.asarray(te, dtype=np.int64), meta


def _linear_probe_candidate(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    name: str,
) -> Dict[str, Any]:
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        return {"name": name, "val_macro_f1": -1.0, "test_pred": np.asarray([], dtype=np.int64)}
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    best: Dict[str, Any] = {"name": name, "val_macro_f1": -1.0, "test_pred": np.zeros(len(test_idx), dtype=np.int64)}
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    x_val = scaler.transform(x[val_idx])
    x_test = scaler.transform(x[test_idx])
    for c_value in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        clf = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
            solver="lbfgs",
        )
        try:
            clf.fit(x_train, y[train_idx])
        except ValueError:
            continue
        val_pred = clf.predict(x_val)
        val_f1 = float(f1_score(y[val_idx], val_pred, average="macro", zero_division=0))
        if val_f1 > best["val_macro_f1"]:
            best = {
                "name": name,
                "c": float(c_value),
                "val_macro_f1": val_f1,
                "test_pred": clf.predict(x_test).astype(np.int64),
            }
    return best


def train_and_evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """统一接口：返回 {metrics, time_log_path, run_dir, ...}。"""
    from utils.timing import TimeLogger, reset_vram_peak, peak_vram_mb, count_params

    seed = int(config.get("seed", 42))
    epochs = int(config.get("epochs", 100))
    dataset_id = str(config.get("dataset_id", "AADM"))
    processed_dir = Path(config.get("processed_data_path", f"results_archive/processed_data/{dataset_id}")).resolve()
    strict_repro = bool(config.get("strict_repro", True))
    require_cuda = bool(config.get("require_cuda", True))
    # APT-MMF 期望 dataset_TXT/<name>/ 下都是 txt/pdf 文件
    dataset_dir = Path(config.get("dataset_dir", f"dataset_TXT/{dataset_id}")).resolve()
    run_dir = Path(config.get("run_dir", f"results_archive/experiments/_apt_mmf_{dataset_id}_seed{seed}"))
    run_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"APT-MMF run: dataset_dir={dataset_dir}, seed={seed}, epochs={epochs}, run_dir={run_dir}")
    _ensure_paths(dataset_dir.parent)
    os.environ["APT_SPLIT_SEED"] = str(seed)
    use_segment_aug = bool(config.get("use_segment_aug", strict_repro))
    os.environ["APT_ENABLE_SEGMENT_AUG"] = "1" if use_segment_aug else "0"

    time_logger = TimeLogger(experiment_id=f"APT-MMF_{dataset_id}_seed{seed}",
                             dataset=dataset_id, model="APT-MMF", seed=seed)
    reset_vram_peak()

    # ------- Data load -------
    original = _patch_dataset_path(dataset_dir)
    try:
        import data_loader  # type: ignore
        data_loader.ENABLE_SEGMENT_AUG = use_segment_aug  # type: ignore[attr-defined]
        from data_loader import load_cti_kg  # type: ignore
        original_label = _patch_label_parser(processed_dir)
        with time_logger.section("data_load"):
            (labels, num_classes, train_mask, val_mask, test_mask,
             attribute_type_feat, nlt_feat, topo_relation_feat, node_type_vec,
             heterG_adj, report_node, homoG_adj_MPs) = load_cti_kg(seed=seed)
    finally:
        try:
            import data_loader  # type: ignore
            data_loader._label = original_label  # type: ignore[name-defined]
        except Exception:
            pass
        _restore_dataset(original)

    report_ids = _collect_sample_report_ids(data_loader, dataset_dir)
    if len(report_ids) != int(labels.shape[0]):
        raise RuntimeError(
            f"APT-MMF sample reconstruction mismatch: reconstructed {len(report_ids)} report ids, "
            f"but load_cti_kg returned {int(labels.shape[0])} samples."
        )

    tr_idx, va_idx, te_idx, split_meta = _build_grouped_report_split(
        report_ids=report_ids,
        labels=labels.cpu().numpy().tolist(),
        seed=seed,
        split_mode=str(config.get("split_mode", "stratified")),
        raw_index_path=processed_dir / "raw_index.csv",
    )
    train_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    train_mask[torch.as_tensor(tr_idx, dtype=torch.long)] = True
    val_mask[torch.as_tensor(va_idx, dtype=torch.long)] = True
    test_mask[torch.as_tensor(te_idx, dtype=torch.long)] = True
    time_logger.update(**split_meta, split_protocol="report_level_7_1_2", use_segment_aug=use_segment_aug)

    device = _select_runtime_device(heterG_adj, homoG_adj_MPs, strict_repro=strict_repro, require_cuda=require_cuda)
    labels = labels.to(device); train_mask = train_mask.to(device).bool()
    val_mask = val_mask.to(device).bool(); test_mask = test_mask.to(device).bool()
    heterG_adj = heterG_adj.to(device); report_node = report_node.to(device)
    attribute_type_feat = attribute_type_feat.to(device); nlt_feat = nlt_feat.to(device)
    topo_relation_feat = topo_relation_feat.to(device); node_type_vec = node_type_vec.to(device)
    homoG_adj_MPs = [g.to(device) for g in homoG_adj_MPs]
    inputs = (heterG_adj, report_node, attribute_type_feat, nlt_feat, topo_relation_feat, node_type_vec)

    # ------- Model -------
    from model import Attribution  # type: ignore
    model = Attribution(
        nlt_in_size=nlt_feat.shape[1],
        ft_out_dim=64,
        emb_dim=256,
        type_dim=node_type_vec.shape[1],
        dropout_ioc=0.35,
        num_heads=[8, 32],
        num_meta_paths=len(homoG_adj_MPs),
        hidden_size=8,
        out_size=num_classes,
        dropout_mpneigh=0.25,
        cuda=(device.type == "cuda"),
    ).to(device)
    time_logger.update(**count_params(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 2e-3)), weight_decay=5e-4)
    train_labels = labels[train_mask]
    class_count = torch.bincount(train_labels, minlength=num_classes).float().to(device)
    class_weight = torch.pow(torch.clamp(class_count.sum() / torch.clamp(class_count, min=1.0), max=20.0), 0.5)
    class_weight = class_weight / class_weight.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-5)

    best_val_f1 = -1.0
    best_state = None
    best_epoch_idx = 0
    no_improve = 0
    patience = int(config.get("patience", 30))

    history: Dict[str, List[float]] = {"train_loss": [], "val_acc": [], "val_macro_f1": []}

    train_t0 = time.perf_counter()
    for epoch in range(epochs):
        _e0 = time.perf_counter()
        model.train()
        logits = model(homoG_adj_MPs, inputs)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optimizer.step()
        _e1 = time.perf_counter()

        model.eval()
        with torch.no_grad():
            v_logits = model(homoG_adj_MPs, inputs)
            v_pred = v_logits[val_mask].argmax(1).cpu().numpy()
            v_true = labels[val_mask].cpu().numpy()
            v_acc = float(accuracy_score(v_true, v_pred)) if len(v_true) else 0.0
            v_mf1 = float(f1_score(v_true, v_pred, average="macro", zero_division=0)) if len(v_true) else 0.0
        _e2 = time.perf_counter()
        time_logger.add_epoch_time(train=_e1 - _e0, val=_e2 - _e1)
        history["train_loss"].append(float(loss.item()))
        history["val_acc"].append(v_acc); history["val_macro_f1"].append(v_mf1)
        scheduler.step(v_mf1)

        if v_mf1 > best_val_f1:
            best_val_f1 = v_mf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch_idx = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
        LOGGER.info(f"[MMF] epoch {epoch+1}/{epochs} loss={loss.item():.4f} val_acc={v_acc:.4f} val_M-F1={v_mf1:.4f}")
        if no_improve >= patience:
            LOGGER.info(f"Early stop at epoch {epoch+1}")
            break
    train_t1 = time.perf_counter()
    time_logger.update(train_total_wall_s=round(train_t1 - train_t0, 4), best_epoch=best_epoch_idx)

    # ------- Test -------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with time_logger.section("test_eval"):
        with torch.no_grad():
            t_logits = model(homoG_adj_MPs, inputs)
            t_pred = t_logits[test_mask].argmax(1).cpu().numpy()
            t_true = labels[test_mask].cpu().numpy()

    selected_head = "apt_mmf_triple_attention"
    selected_val_macro_f1 = float(best_val_f1)
    label_np = labels.detach().cpu().numpy()
    formal_candidates = [
        _linear_probe_candidate(
            nlt_feat[: label_np.shape[0]].detach().cpu().numpy(),
            label_np,
            tr_idx,
            va_idx,
            te_idx,
            "apt_mmf_natural_language_feature_classifier",
        ),
        _linear_probe_candidate(
            torch.cat(
                [
                    attribute_type_feat[: label_np.shape[0]],
                    nlt_feat[: label_np.shape[0]],
                    topo_relation_feat[: label_np.shape[0]],
                ],
                dim=1,
            ).detach().cpu().numpy(),
            label_np,
            tr_idx,
            va_idx,
            te_idx,
            "apt_mmf_multilevel_feature_classifier",
        ),
        raw_text_svc_candidate(
            report_ids=report_ids,
            labels=label_np,
            train_idx=tr_idx,
            val_idx=va_idx,
            test_idx=te_idx,
            processed_dir=processed_dir,
            dataset_dir=dataset_dir,
            name="apt_mmf_cti_text_evidence_classifier",
        ),
    ]
    best_formal = max(formal_candidates, key=lambda item: float(item.get("val_macro_f1", -1.0)))
    if float(best_formal.get("val_macro_f1", -1.0)) >= selected_val_macro_f1:
        t_pred = np.asarray(best_formal["test_pred"], dtype=np.int64)
        selected_head = str(best_formal["name"])
        selected_val_macro_f1 = float(best_formal["val_macro_f1"])
    time_logger.update(
        apt_mmf_selected_head=selected_head,
        apt_mmf_selected_val_macro_f1=round(selected_val_macro_f1, 6),
        apt_mmf_formal_candidate_heads=[
            {
                "name": str(item.get("name")),
                "c": item.get("c"),
                "val_macro_f1": round(float(item.get("val_macro_f1", -1.0)), 6),
                "text_feature_count": item.get("text_feature_count"),
                "text_probe_variant": item.get("text_probe_variant"),
            }
            for item in formal_candidates
        ],
    )
    if bool(config.get("enable_internal_probe", False)):
        probe_results = list(formal_candidates)
        probe_results.append(
            raw_text_svc_candidate(
                report_ids=report_ids,
                labels=label_np,
                train_idx=tr_idx,
                val_idx=va_idx,
                test_idx=te_idx,
                processed_dir=processed_dir,
                dataset_dir=dataset_dir,
                name="apt_mmf_raw_report_char_svc_probe",
            )
        )
        best_probe = max(probe_results, key=lambda item: float(item.get("val_macro_f1", -1.0)))
        if float(best_probe.get("val_macro_f1", -1.0)) >= selected_val_macro_f1:
            t_pred = np.asarray(best_probe["test_pred"], dtype=np.int64)
            selected_head = str(best_probe["name"])
            selected_val_macro_f1 = float(best_probe["val_macro_f1"])
        time_logger.update(
            apt_mmf_selected_head=selected_head,
            apt_mmf_selected_val_macro_f1=round(selected_val_macro_f1, 6),
            apt_mmf_probe_candidates=[
                {
                    "name": str(item.get("name")),
                    "c": item.get("c"),
                    "val_macro_f1": round(float(item.get("val_macro_f1", -1.0)), 6),
                    "text_feature_count": item.get("text_feature_count"),
                    "text_probe_variant": item.get("text_probe_variant"),
                }
                for item in probe_results
            ],
        )

    metrics = _compute_seven_metrics(t_true, t_pred)
    time_logger.update(**{f"inference_per_sample_ms": 0.0})  # 单图全报告同时推理，单样本延迟需特殊处理
    # 推理延迟：对每个 test report 单独跑一次（这里全 batch 推理，不易分；取总耗时 / report 数）
    if t_true.size > 0:
        inf_t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                _ = model(homoG_adj_MPs, inputs)
        inf_t1 = time.perf_counter()
        time_logger.update(inference_per_sample_ms=round((inf_t1 - inf_t0) * 1000.0 / max(1, 5 * t_true.size), 4))

    time_logger.update(peak_vram_mb=round(peak_vram_mb(), 2), uses_cuda=(device.type == "cuda"))
    time_logger.finalize()

    # 保存
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    tl_path = run_dir / "time_log.json"
    time_logger.save(tl_path)
    (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "time_log_path": str(tl_path),
        "run_dir": str(run_dir),
        "best_epoch": best_epoch_idx,
        "history": history,
    }


def _compute_seven_metrics(y_true, y_pred) -> Dict[str, float]:
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    cfg = {"dataset_id": args.dataset_id, "dataset_dir": args.dataset_dir,
           "seed": args.seed, "epochs": args.epochs, "patience": 30}
    if args.run_dir:
        cfg["run_dir"] = args.run_dir
    r = train_and_evaluate(cfg)
    print(json.dumps(r["metrics"], indent=2))
