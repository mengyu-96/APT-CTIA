from __future__ import annotations

import argparse
import importlib.util
import csv
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.aggregate_results import aggregate_experiment_tree
from utils.runner import run_seeds_parallel, run_seeds_sequential


LOGGER = logging.getLogger("experiment_suite")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

GRAPH_MODELS = {"RGAT", "GAT", "GCN", "HGT", "Transformer", "GraphSAGE", "GIN", "Hybrid", "APT-KGL-Rep"}
GRAPH_MODEL_ALIASES = {"SAPCL-Approx": "RGAT"}

SUPPORTED_MODELS = {
    *GRAPH_MODELS,
    *GRAPH_MODEL_ALIASES,
    "APT-ATT",
    "APT-MMF",
    "Mead",
    "MLDSJ",
    "TRAIL",
}

REQUIRED_METRIC_FIELDS = [
    "accuracy",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]

REQUIRED_TIME_LOG_FIELDS = [
    "experiment_id",
    "dataset",
    "model",
    "seed",
    "start_ts",
    "end_ts",
    "wall_clock_total_s",
    "t_train_total_s",
    "t_val_total_s",
    "inference_per_sample_ms",
    "peak_vram_mb",
    "peak_ram_mb",
    "experiment_date",
    "timezone",
    "timezone_name",
]

DATASET_MAP = {
    "AADM": {
        "dataset_id": "AADM",
        "dataset_dir": ROOT / "dataset_TXT" / "AADM",
        "processed_data_root": ROOT / "results_archive" / "processed_data",
        "min_class_samples": 1,
        "only_txt": True,
    },
    "APT-Notes": {
        "dataset_id": "APT-Notes",
        "dataset_dir": ROOT / "dataset_TXT" / "APT-Notes",
        "processed_data_root": ROOT / "results_archive" / "processed_data",
        "min_class_samples": 1,
        "only_txt": False,
    },
    "APT-CTI": {
        "dataset_id": "APT-CTI",
        "dataset_dir": ROOT / "dataset_TXT" / "apt_groups",
        "processed_data_root": ROOT / "results_archive" / "processed_data",
        "min_class_samples": 1,
        "only_txt": False,
    },
}

BASELINE_METHODS = {
    "APT-ATT": "APT-ATT",
    "APT-MMF": "APT-MMF",
    "Mead": "Mead",
    "TRAIL": "TRAIL",
    "MLDSJ": "MLDSJ",
}


def _method_for_model(model: str) -> str:
    if model == "SAPCL-Approx":
        return "SAPCL-Approx"
    if model in GRAPH_MODELS:
        return "GRACE"
    return BASELINE_METHODS.get(model, model)


def _safe_path_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()) or "default"


def _variant_output_pattern(rq: str, variant_label: Optional[str]) -> str:
    if variant_label:
        return f"{rq}/<dataset>/<fixed_downstream_model>/{_safe_path_segment(variant_label)}/seed_<seed>"
    return f"{rq}/<dataset>/<method_or_model>/seed_<seed>"


def _variant_change_summary(args: argparse.Namespace) -> Dict[str, Any]:
    use_temperature = not bool(getattr(args, "disable_temperature_calibration", False))
    return {
        "ablation_mode": getattr(args, "ablation_mode", "dual"),
        "graph_variant": getattr(args, "graph_variant", "full"),
        "feature_variant": getattr(args, "feature_variant", "full"),
        "fusion_mode": getattr(args, "fusion_mode", "adaptive"),
        "pooling_mode": getattr(args, "pooling_mode", "attention"),
        "rgcn_num_bases": getattr(args, "rgcn_num_bases", 30),
        "loss_type": getattr(args, "loss_type", None),
        "use_temperature_calibration": use_temperature,
    }


def _write_variant_index(
    base_output_dir: Path,
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    seeds: List[int],
    gpus: List[int],
) -> None:
    rq_dir = base_output_dir / str(args.rq)
    rq_dir.mkdir(parents=True, exist_ok=True)
    variant_label = str(getattr(args, "variant_label", "") or "default")
    index_path = rq_dir / "VARIANTS_INDEX.json"
    md_path = rq_dir / "VARIANTS_INDEX.md"

    existing: Dict[str, Any] = {"rq": args.rq, "variants": []}
    if index_path.exists():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except json.JSONDecodeError:
            existing = {"rq": args.rq, "variants": []}

    variants = [item for item in existing.get("variants", []) if item.get("variant_label") != variant_label]
    entry = {
        "variant_label": variant_label,
        "rq": args.rq,
        "suite_name": args.suite_name,
        "description": getattr(args, "notes", None),
        "datasets": list(args.datasets),
        "fixed_downstream_models": list(args.models),
        "target_changes": _variant_change_summary(args),
        "preprocess_reuse_policy": _preprocess_reuse_policy(args.rq),
        "preprocess_paths": manifest.get("preprocess_paths", {}),
        "seeds": seeds,
        "gpus": gpus,
        "output_pattern": _variant_output_pattern(str(args.rq), None if variant_label == "default" else variant_label),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    variants.append(entry)
    variants.sort(key=lambda item: str(item.get("variant_label", "")))

    index = {
        "rq": args.rq,
        "layout": "<base_output_dir>/<rq>/<dataset>/<fixed_downstream_model>/<variant_label>/seed_<seed>",
        "note": "The model directory is the fixed downstream model used for controlled comparison; variant_label identifies the compared module.",
        "variants": variants,
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {args.rq} Variant Index",
        "",
        "Layout: `<base_output_dir>/<rq>/<dataset>/<fixed_downstream_model>/<variant_label>/seed_<seed>`",
        "",
        "`fixed_downstream_model` is the controlled downstream model, not the variant name. Use `variant_label` to distinguish methods/modules within the same RQ.",
        "",
        "| variant_label | fixed_downstream_models | target_changes | preprocess_reuse_policy | output_pattern |",
        "|---|---|---|---|---|",
    ]
    for item in variants:
        changes = item.get("target_changes", {})
        changed_items = [
            f"{key}={value}"
            for key, value in changes.items()
            if value not in (None, "full", "dual", "adaptive", "attention", 30, True)
        ]
        change_text = "; ".join(changed_items) if changed_items else "baseline/default module settings"
        lines.append(
            "| {label} | {models} | {changes} | {reuse} | `{pattern}` |".format(
                label=item.get("variant_label", ""),
                models=", ".join(str(model) for model in item.get("fixed_downstream_models", [])),
                changes=change_text,
                reuse=item.get("preprocess_reuse_policy", ""),
                pattern=item.get("output_pattern", ""),
            )
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _preprocess_reuse_policy(rq: Optional[str]) -> str:
    rq_name = str(rq or "").strip()
    if rq_name == "RQ3_preprocess":
        return "rq3_variant_isolated"
    return "reuse_rq1_method_preprocess"


def _processed_path_for_model(dataset: str, model: str) -> Path:
    return _processed_path_for_context(dataset, model)


def _processed_path_for_context(
    dataset: str,
    model: str,
    rq: Optional[str] = None,
    variant_label: Optional[str] = None,
) -> Path:
    dataset_info = DATASET_MAP[dataset]
    root = Path(dataset_info["processed_data_root"])
    method = _method_for_model(model)
    if _preprocess_reuse_policy(rq) == "rq3_variant_isolated":
        variant_segment = _safe_path_segment(variant_label or model)
        return root / "RQ3_preprocess" / variant_segment / method / str(dataset_info["dataset_id"])
    return root / method / str(dataset_info["dataset_id"])


def _load_rows_from_raw_index(raw_index_path: Path) -> List[Dict[str, str]]:
    with raw_index_path.open("r", encoding="utf-8", errors="ignore", newline="") as fp:
        return list(csv.DictReader(fp))


def _write_label_mapping_from_rows(rows: List[Dict[str, str]], output_dir: Path) -> None:
    labels = sorted({(row.get("apt_group") or "UNKNOWN").strip() or "UNKNOWN" for row in rows})
    mapping = {label: idx for idx, label in enumerate(labels)}
    (output_dir / "label_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_preprocess_manifest(output_dir: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    (output_dir / "preprocess_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _baseline_index_ready(output_dir: Path) -> bool:
    return (output_dir / "raw_index.csv").exists() and (output_dir / "label_mapping.json").exists()


def _prepare_baseline_index(dataset: str, model: str, output_dir: Path) -> None:
    dataset_info = DATASET_MAP[dataset]
    dataset_dir = Path(dataset_info["dataset_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if _baseline_index_ready(output_dir):
        rows = _load_rows_from_raw_index(output_dir / "raw_index.csv")
    else:
        from preprocess_apt_dataset import build_raw_index

        rows_meta = build_raw_index(
            pdf_root=None,
            txt_root=dataset_dir,
            output_csv=output_dir / "raw_index.csv",
            only_txt=bool(dataset_info.get("only_txt", False)),
        )
        rows = [item.to_row() for item in rows_meta]
        _write_label_mapping_from_rows(rows, output_dir)
    _write_preprocess_manifest(
        output_dir,
        {
            "dataset": dataset,
            "dataset_id": dataset_info["dataset_id"],
            "model": model,
            "method": _method_for_model(model),
            "preprocess_scope": "method_local_index_only",
            "preprocess_reuse_policy": "method_local",
            "source_dataset_dir": str(dataset_dir),
            "report_count": len(rows),
            "uses_apt_ctia_graphs": False,
            "uses_apt_ctia_entities": False,
            "notes": (
                "This directory intentionally contains only raw report index and label mapping. "
                "The baseline implementation rebuilds its paper-specific features/graphs from raw reports."
            ),
        },
    )


def _apt_ctia_graph_ready(output_dir: Path) -> bool:
    required = [output_dir / "raw_index.csv", output_dir / "label_mapping.json", output_dir / "graphs.pt", output_dir / "graph_stats.json"]
    if not all(path.exists() for path in required):
        return False
    try:
        stats = json.loads((output_dir / "graph_stats.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(stats, list) or not stats:
        return False
    return all(bool(item.get("use_text_embedding")) and int(item.get("embedding_dim", 0)) == 384 for item in stats)


def _rq3_scaffold_ready(output_dir: Path) -> bool:
    required = [
        output_dir / "raw_index.csv",
        output_dir / "label_mapping.json",
        output_dir / "paragraphs.jsonl",
        output_dir / "entities.jsonl",
        output_dir / "graphs.pt",
        output_dir / "preprocess_manifest.json",
    ]
    if not all(path.exists() for path in required):
        return False
    manifest = _load_json_if_exists(output_dir / "preprocess_manifest.json")
    return (
        manifest.get("preprocess_scope") == "rq3_module_scaffold"
        and manifest.get("preprocess_reuse_policy") == "rq3_variant_isolated"
        and manifest.get("uses_apt_ctia_graphs") is False
    )


def _read_report_text_for_rq3(path: Path) -> str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="ignore")
        except Exception:
            continue
    return ""


def _resolve_raw_report_path(row: Dict[str, str], dataset_dir: Path) -> Path:
    candidates: List[Path] = []
    raw_path = Path(str(row.get("file_path", "")).strip())
    if str(raw_path).strip():
        candidates.append(raw_path)
        candidates.append(dataset_dir / raw_path.name)
    raw_name = str(row.get("raw_name", "")).strip()
    file_type = str(row.get("file_type", "")).strip()
    if raw_name:
        candidates.append(dataset_dir / f"{raw_name}{file_type}")
    report_id = str(row.get("report_id", "")).strip()
    if report_id:
        candidates.append(dataset_dir / f"{report_id}{file_type}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw_path


_RQ3_ENTITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("URL", re.compile(r"https?://[^\s)>\"]+", re.IGNORECASE)),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("CVE", re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)),
    ("MITRE_TECH", re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)),
    ("HASH_SHA256", re.compile(r"\b[a-fA-F0-9]{64}\b")),
    ("HASH_SHA1", re.compile(r"\b[a-fA-F0-9]{40}\b")),
    ("HASH_MD5", re.compile(r"\b[a-fA-F0-9]{32}\b")),
    ("DOMAIN", re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")),
    ("FILE_PATH", re.compile(r"(?:[A-Za-z]:\\\\|/)[^\s,;:]+")),
    ("REGISTRY", re.compile(r"\bHK(?:LM|CU|CR|U|CC)\\\\[^\s,;]+", re.IGNORECASE)),
)


def _extract_rq3_entities(text: str, report_id: str, paragraph_index: int) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for label, pattern in _RQ3_ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip().rstrip(".,;:")
            if not value:
                continue
            key = (label, match.start(), match.start() + len(value))
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                {
                    "report_id": report_id,
                    "paragraph_index": int(paragraph_index),
                    "label": label,
                    "text": value,
                    "start": int(match.start()),
                    "end": int(match.start() + len(value)),
                }
            )
    return sorted(entities, key=lambda item: (int(item["start"]), str(item["label"])))


def _prepare_rq3_module_scaffold(dataset: str, model: str, output_dir: Path, variant_label: str) -> None:
    if _rq3_scaffold_ready(output_dir):
        return

    import torch
    from torch_geometric.data import Data

    dataset_info = DATASET_MAP[dataset]
    dataset_dir = Path(dataset_info["dataset_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    from preprocess_apt_dataset import build_raw_index

    rows_meta = build_raw_index(
        pdf_root=None,
        txt_root=dataset_dir,
        output_csv=output_dir / "raw_index.csv",
        only_txt=bool(dataset_info.get("only_txt", False)),
    )
    rows = [item.to_row() for item in rows_meta]
    _write_label_mapping_from_rows(rows, output_dir)
    label_map = json.loads((output_dir / "label_mapping.json").read_text(encoding="utf-8"))

    paragraphs_written = 0
    entities_written = 0
    graphs = []
    feature_dim = 30 + 64 + 64 + 64 + 3 + 384 + 300
    with (output_dir / "paragraphs.jsonl").open("w", encoding="utf-8") as para_fp, (
        output_dir / "entities.jsonl"
    ).open("w", encoding="utf-8") as ent_fp:
        for row in rows:
            report_id = str(row.get("report_id") or row.get("raw_name") or "").strip()
            label_name = str(row.get("apt_group") or "UNKNOWN").strip() or "UNKNOWN"
            if not report_id or label_name not in label_map:
                continue
            report_path = _resolve_raw_report_path(row, dataset_dir)
            text = _read_report_text_for_rq3(report_path)
            if not text.strip():
                continue
            paragraphs = [part.strip() for part in re.split(r"\n{2,}|(?<=[.!?])\s+", text) if len(part.strip()) >= 20]
            if not paragraphs:
                paragraphs = [text.strip()]
            for para_idx, paragraph in enumerate(paragraphs):
                para_record = {"report_id": report_id, "paragraph_index": int(para_idx), "text": paragraph}
                para_fp.write(json.dumps(para_record, ensure_ascii=False) + "\n")
                paragraphs_written += 1
                for entity in _extract_rq3_entities(paragraph, report_id, para_idx):
                    ent_fp.write(json.dumps(entity, ensure_ascii=False) + "\n")
                    entities_written += 1
            graph = Data(
                x=torch.zeros((1, feature_dim), dtype=torch.float),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                y=torch.tensor([int(label_map[label_name])], dtype=torch.long),
            )
            graph.report_id = report_id
            graph.apt_group = label_name
            graph.node_texts = ["REPORT"]
            graph.node_labels = ["REPORT"]
            graph.node_paragraph_indices = [-1]
            graph.doc_emb = torch.zeros((1, 384), dtype=torch.float)
            graphs.append(graph)
    torch.save(graphs, output_dir / "graphs.pt")
    _write_preprocess_manifest(
        output_dir,
        {
            "dataset": dataset,
            "dataset_id": dataset_info["dataset_id"],
            "model": model,
            "method": _method_for_model(model),
            "rq_context": "RQ3_preprocess",
            "variant_label": variant_label,
            "preprocess_scope": "rq3_module_scaffold",
            "preprocess_reuse_policy": "rq3_variant_isolated",
            "reusable_from_rq1": False,
            "source_dataset_dir": str(dataset_dir),
            "report_count": len(graphs),
            "paragraph_count": paragraphs_written,
            "entity_count": entities_written,
            "uses_apt_ctia_graphs": False,
            "uses_apt_ctia_minilm_features": False,
            "notes": (
                "RQ3 scaffold built from raw reports only. It intentionally avoids the RQ1 APT-CTIA graph builder; "
                "the selected feature_variant reconstructs the compared preprocessing module before RGAT training."
            ),
        },
    )


def _prepare_apt_ctia_graphs(dataset: str, output_dir: Path) -> None:
    dataset_info = DATASET_MAP[dataset]
    if _apt_ctia_graph_ready(output_dir):
        return

    from backend.core.repro_paths import LOCAL_MINILM_ROOT
    from preprocess_apt_dataset import DatasetPreprocessor, GraphBuilderConfig, GraphDatasetBuilder, PreprocessConfig

    if not LOCAL_MINILM_ROOT.exists():
        raise FileNotFoundError(
            f"GRACE strict preprocessing requires local all-MiniLM-L6-v2 checkpoint: {LOCAL_MINILM_ROOT}"
        )

    dataset_dir = Path(dataset_info["dataset_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = DatasetPreprocessor(
        PreprocessConfig(
            pdf_root=None,
            txt_root=dataset_dir,
            output_dir=output_dir,
            min_paragraph_length=20,
            skip_empty=True,
            only_txt=bool(dataset_info.get("only_txt", False)),
        )
    )
    preprocessor.run()
    graph_config = GraphBuilderConfig(
        min_entities_per_graph=3,
        feature_dim=64,
        add_report_node=True,
        use_text_embedding=True,
        embedding_model=str(LOCAL_MINILM_ROOT),
        embed_context=True,
        context_window_size=64,
        use_tfidf=True,
        tfidf_dim=300,
    )
    builder = GraphDatasetBuilder(
        metadata_path=preprocessor.metadata_path,
        entities_path=preprocessor.entities_path,
        output_dir=output_dir,
        config=graph_config,
        paragraphs_path=preprocessor.paragraphs_path,
    )
    builder.run()
    if not _apt_ctia_graph_ready(output_dir):
        raise RuntimeError(
            f"GRACE preprocessing completed but MiniLM graph features were not verified in {output_dir}."
        )
    _write_preprocess_manifest(
        output_dir,
        {
            "dataset": dataset,
            "dataset_id": dataset_info["dataset_id"],
            "model_family": "GRACE",
            "method": "GRACE",
            "preprocess_scope": "apt_ctia_full_graph",
            "preprocess_reuse_policy": "reuse_allowed_for_rq2_rq4_rq5",
            "source_dataset_dir": str(dataset_dir),
            "uses_apt_ctia_graphs": True,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_model_path": str(LOCAL_MINILM_ROOT),
            "embedding_dim": 384,
            "embed_context": True,
            "context_window_size": 64,
            "use_tfidf": True,
            "tfidf_dim": 300,
        },
    )


def _prepare_sapcl_approx_graphs(dataset: str, output_dir: Path) -> None:
    _prepare_apt_ctia_graphs(dataset, output_dir)
    manifest_path = output_dir / "preprocess_manifest.json"
    manifest = _load_json_if_exists(manifest_path)
    manifest.update(
        {
            "model_family": "SAPCL-Approx",
            "method": "SAPCL-Approx",
            "preprocess_scope": "syntax_aware_approx_graph",
            "approximate_reproduction": True,
            "strict_paper_reproduction": False,
            "uses_apt_ctia_graphs": False,
            "notes": (
                "Approximate Syntax-aware graph network / SAPCL preprocessing. "
                "It uses local report text, entity spans, type rules, trigger terms, "
                "and short-window constraints to build syntax/semantic candidate edges. "
                "It is not the supervised BERT+SA-GAT+prototype-contrastive+decoder reproduction."
            ),
        }
    )
    _write_preprocess_manifest(output_dir, manifest)


def ensure_processed_data(
    dataset: str,
    model: str,
    dry_run: bool = False,
    rq: Optional[str] = None,
    variant_label: Optional[str] = None,
) -> Path:
    output_dir = _processed_path_for_context(dataset, model, rq=rq, variant_label=variant_label)
    if dry_run:
        return output_dir
    if _preprocess_reuse_policy(rq) == "rq3_variant_isolated" and model in GRAPH_MODELS | set(GRAPH_MODEL_ALIASES):
        _prepare_rq3_module_scaffold(dataset, model, output_dir, variant_label or model)
    elif model == "SAPCL-Approx":
        _prepare_sapcl_approx_graphs(dataset, output_dir)
    elif model in GRAPH_MODELS:
        _prepare_apt_ctia_graphs(dataset, output_dir)
    else:
        _prepare_baseline_index(dataset, model, output_dir)
    manifest_path = output_dir / "preprocess_manifest.json"
    manifest = _load_json_if_exists(manifest_path)
    manifest.update(
        {
            "rq_context": rq,
            "variant_label": variant_label,
            "preprocess_reuse_policy": _preprocess_reuse_policy(rq),
            "reusable_from_rq1": _preprocess_reuse_policy(rq) != "rq3_variant_isolated",
        }
    )
    _write_preprocess_manifest(output_dir, manifest)
    return output_dir


def parse_seeds(seed_spec: str) -> List[int]:
    if "," in seed_spec:
        return [int(item.strip()) for item in seed_spec.split(",") if item.strip()]
    if ":" in seed_spec:
        start_s, end_s = seed_spec.split(":", 1)
        start = int(start_s)
        end = int(end_s)
        return list(range(start, end + 1))
    return [int(seed_spec)]


def parse_gpus(gpu_spec: str | None) -> Optional[List[int]]:
    if gpu_spec is None:
        return None
    gpu_spec = gpu_spec.strip()
    if not gpu_spec:
        return []
    return [int(item.strip()) for item in gpu_spec.split(",") if item.strip()]


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            sanitized[key] = str(value)
        else:
            sanitized[key] = value
    return sanitized


def _build_run_dir(base_output_dir: Path, rq: str, dataset: str, model: str, seed: int) -> Path:
    return base_output_dir / rq / dataset / model / f"seed_{seed}"


def _sanitize_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "default"


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _missing_required_fields(payload: Dict[str, Any], fields: List[str]) -> List[str]:
    return [field for field in fields if field not in payload or payload.get(field) in (None, "")]


def _validate_run_artifacts(run_dir: Path, metrics: Dict[str, Any], time_log: Dict[str, Any]) -> None:
    missing_metrics = _missing_required_fields(metrics, REQUIRED_METRIC_FIELDS)
    missing_time = _missing_required_fields(time_log, REQUIRED_TIME_LOG_FIELDS)
    if missing_metrics or missing_time:
        raise RuntimeError(
            "Experiment run did not produce complete formal artifacts: "
            f"run_dir={run_dir}, missing_metrics={missing_metrics}, missing_time_log={missing_time}"
        )


def _dispatch_model(model: str, config: Dict[str, Any]) -> Dict[str, Any]:
    if model in GRAPH_MODEL_ALIASES:
        model_config = dict(config)
        model_config["model_type"] = GRAPH_MODEL_ALIASES[model]
        model_config["reported_model_type"] = model
        from backend.core.train import run_training_pipeline

        return run_training_pipeline(model_config)
    if model in GRAPH_MODELS:
        from backend.core.train import run_training_pipeline

        model_config = dict(config)
        model_config["model_type"] = model
        return run_training_pipeline(model_config)
    if model == "APT-ATT":
        from backend.core.baselines.apt_att import train_and_evaluate as run_apt_att

        return run_apt_att(config)
    if model == "APT-MMF":
        from backend.core.baselines.apt_mmf import train_and_evaluate as run_apt_mmf

        return run_apt_mmf(config)
    if model == "Mead":
        from backend.core.baselines.mead import train_and_evaluate as run_mead

        return run_mead(config)
    if model == "MLDSJ":
        from backend.core.baselines.mldsj import train_and_evaluate as run_mldsj

        return run_mldsj(config)
    if model == "TRAIL":
        from backend.core.baselines.trail import train_and_evaluate as run_trail

        return run_trail(config)
    raise ValueError(f"Unsupported model: {model}")


def suite_worker(seed: int, gpu_id: Optional[int], config: Dict[str, Any]) -> Dict[str, Any]:
    rq = str(config["rq"])
    suite_name = str(config["suite_name"])
    dataset = str(config["dataset"])
    model = str(config["model"])
    raw_variant_label = config.get("variant_label", "")
    variant_label = "" if raw_variant_label is None else str(raw_variant_label).strip()
    base_output_dir = Path(config["base_output_dir"])
    run_dir = _build_run_dir(base_output_dir, rq, dataset, model, int(seed))
    if variant_label:
        run_dir = run_dir.parent / _sanitize_segment(variant_label) / run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = DATASET_MAP[dataset]
    processed_data_path = ensure_processed_data(
        dataset,
        model,
        dry_run=False,
        rq=rq,
        variant_label=variant_label,
    )
    min_class_samples = config.get("min_class_samples", None)
    if min_class_samples is None:
        min_class_samples = dataset_info.get("min_class_samples", 1)
    run_config = {
        "seed": int(seed),
        "gpu_id": -1 if gpu_id is None else int(gpu_id),
        "dataset_id": dataset_info["dataset_id"],
        "dataset_dir": dataset_info["dataset_dir"],
        "processed_data_path": processed_data_path,
        "preprocess_method": _method_for_model(model),
        "preprocess_reuse_policy": _preprocess_reuse_policy(rq),
        "min_class_samples": int(min_class_samples),
        "run_dir": run_dir,
        "epochs": int(config.get("epochs", 100)),
        "lr": float(config.get("lr", 0.001)),
        "batch_size": int(config.get("batch_size", 32)),
        "ablation_mode": config.get("ablation_mode", "dual"),
        "use_ctgan": bool(config.get("use_ctgan", False)),
        "multi_gpu_dp": bool(config.get("multi_gpu_dp", False)),
        "use_compile": bool(config.get("use_compile", False)),
        "graph_variant": config.get("graph_variant", "full"),
        "feature_variant": config.get("feature_variant", "full"),
        "split_mode": config.get("split_mode", "stratified"),
        "fusion_mode": config.get("fusion_mode", "adaptive"),
        "pooling_mode": config.get("pooling_mode", "attention"),
        "rgcn_num_bases": config.get("rgcn_num_bases", 30),
        "loss_type": config.get("loss_type", None),
        "focal_gamma": config.get("focal_gamma", 2.0),
        "use_temperature_calibration": bool(config.get("use_temperature_calibration", True)),
        "strict_repro": bool(config.get("strict_repro", True)),
        "require_cuda": bool(config.get("require_cuda", True)),
        "enable_internal_probe": bool(config.get("enable_internal_probe", False)),
        "variant_label": variant_label,
        "notes": config.get("notes"),
    }

    result = _dispatch_model(model, run_config)

    metrics = _load_json_if_exists(run_dir / "metrics.json")
    if not metrics and isinstance(result.get("metrics"), dict):
        metrics = result["metrics"]

    time_log = _load_json_if_exists(run_dir / "time_log.json")
    results_json = _load_json_if_exists(run_dir / "results.json")
    _validate_run_artifacts(run_dir, metrics, time_log)

    record = {
        "suite_name": suite_name,
        "rq": rq,
        "dataset": dataset,
        "model": model,
        "variant_label": variant_label,
        "seed": int(seed),
        "status": "ok",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "processed_data_path": str(processed_data_path),
        "preprocess_method": _method_for_model(model),
        "preprocess_reuse_policy": _preprocess_reuse_policy(rq),
        "graph_variant": run_config["graph_variant"],
        "feature_variant": run_config["feature_variant"],
        "ablation_mode": run_config["ablation_mode"],
        "fusion_mode": run_config["fusion_mode"],
        "pooling_mode": run_config["pooling_mode"],
        "rgcn_num_bases": run_config["rgcn_num_bases"],
        "loss_type": run_config["loss_type"],
        "use_temperature_calibration": run_config["use_temperature_calibration"],
        "output_layout": f"{rq}/{dataset}/{model}/{variant_label or 'default'}/seed_{int(seed)}",
        "artifacts": {
            "metrics_path": str(run_dir / "metrics.json"),
            "time_log_path": str(run_dir / "time_log.json"),
            "results_path": str(run_dir / "results.json"),
        },
        "config": _sanitize_config(run_config),
        "metrics": metrics,
        "time_log": time_log,
        "results": results_json,
    }
    with (run_dir / "experiment_record.json").open("w", encoding="utf-8") as fp:
        json.dump(record, fp, ensure_ascii=False, indent=2, default=str)
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "time_log_path": str(run_dir / "time_log.json"),
        "record_path": str(run_dir / "experiment_record.json"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified experiment suite runner for APT attribution experiments.")
    parser.add_argument("--rq", default="RQ1", help="Experiment family, e.g. RQ1/RQ2/RQ3.")
    parser.add_argument("--suite-name", default="APT-DSHG")
    parser.add_argument("--datasets", nargs="+", default=["AADM", "APT-Notes"], choices=sorted(DATASET_MAP.keys()))
    parser.add_argument("--models", nargs="+", default=["RGAT", "APT-ATT", "APT-MMF", "Mead", "TRAIL"])
    parser.add_argument("--seeds", default="42", help="Single seed, comma list, or range like 42:46.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-class-samples", type=int, default=None, help="Drop labels with fewer reports before splitting; default is dataset-specific.")
    parser.add_argument("--ablation-mode", default="dual")
    parser.add_argument("--graph-variant", default="full", help="Graph transform variant, e.g. full/no_bridge_edges/no_root_node/paragraph_cooccurrence/sliding_window_cooccurrence.")
    parser.add_argument("--feature-variant", default="full", help="Feature transform variant, e.g. full/knowledge_extraction_framework.")
    parser.add_argument("--split-mode", default="stratified", choices=["stratified", "time_based"], help="Report-level split mode.")
    parser.add_argument("--fusion-mode", default="adaptive", help="RGAT fusion mode: adaptive or mean.")
    parser.add_argument("--pooling-mode", default="attention", help="RGAT graph readout: attention or mean.")
    parser.add_argument("--rgcn-num-bases", type=int, default=30, help="RGCN basis count; use 0 to disable basis decomposition.")
    parser.add_argument("--loss-type", default=None, help="Training loss override, e.g. focal or cross_entropy.")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--disable-temperature-calibration", action="store_true", help="Force T=1 for confidence calibration ablation.")
    parser.add_argument("--use-ctgan", action="store_true", help="Enable CTGAN augmentation for the APT-ATT baseline.")
    parser.add_argument("--base-output-dir", default=str(ROOT / "results_archive" / "experiments" / "APT-DSHG"))
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids. Empty string forces CPU.")
    parser.add_argument("--parallel", action="store_true", help="Run multi-seed jobs in parallel using utils.runner.")
    parser.add_argument("--multi-gpu-dp", action="store_true")
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback. By default formal experiments require CUDA.")
    parser.add_argument("--allow-approximate", action="store_true", help="Allow approximate/non-verified reproductions to run.")
    parser.add_argument("--enable-diagnostic-probes", action="store_true", help="Enable non-paper diagnostic candidate heads; never use for formal strict RQ1 tables.")
    parser.add_argument("--variant-label", default=None, help="Logical variant name stored in manifests and run records.")
    parser.add_argument("--notes", default=None, help="Optional notes to attach to manifests and run records.")
    parser.add_argument("--dry-run", action="store_true", help="Only emit the suite plan and manifests; do not execute training.")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    unsupported = [model for model in args.models if model not in SUPPORTED_MODELS]
    if unsupported:
        raise ValueError(f"Unsupported models: {unsupported}. Supported models: {sorted(SUPPORTED_MODELS)}")

    if bool(getattr(args, "dry_run", False)) or bool(getattr(args, "aggregate_only", False)):
        return

    if not bool(getattr(args, "allow_cpu", False)):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch import failed while verifying CUDA readiness.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Formal experiment execution requires CUDA, but torch.cuda.is_available() is False in the current runtime. "
                "If you intentionally want CPU fallback for debugging only, re-run with --allow-cpu."
            )

    missing: List[str] = []
    for dataset in args.datasets:
        info = DATASET_MAP[dataset]
        if "dataset_dir" in info and not Path(info["dataset_dir"]).exists():
            missing.append(f"{dataset}: missing dataset_dir {info['dataset_dir']}")
    if missing:
        raise FileNotFoundError("Dataset prerequisites missing:\n" + "\n".join(missing))

    strict_repro = not bool(getattr(args, "allow_approximate", False))
    if strict_repro:
        dep_errors: List[str] = []
        model_set = set(args.models)
        if "APT-ATT" in model_set:
            if importlib.util.find_spec("nltk") is None:
                dep_errors.append("APT-ATT strict reproduction requires `nltk` in env `GRACE`.")
            if importlib.util.find_spec("xgboost") is None:
                dep_errors.append("APT-ATT strict reproduction requires `xgboost` in env `GRACE`.")
            if bool(getattr(args, "use_ctgan", False)) and importlib.util.find_spec("ctgan") is None:
                dep_errors.append("APT-ATT strict reproduction with CTGAN enabled requires `ctgan` in env `GRACE`.")
        if "APT-MMF" in model_set and importlib.util.find_spec("dgl") is None:
            dep_errors.append("APT-MMF strict reproduction requires `dgl` in env `GRACE`.")
        if "TRAIL" in model_set and importlib.util.find_spec("torch_geometric") is None:
            dep_errors.append("TRAIL strict reproduction requires `torch_geometric` in env `GRACE`.")
        if dep_errors:
            raise RuntimeError("Strict reproduction prerequisites missing:\n" + "\n".join(dep_errors))


def run_suite(args: argparse.Namespace) -> Dict[str, Any]:
    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        reports_dir = base_output_dir / "_reports"
        outputs = aggregate_experiment_tree(base_output_dir, reports_dir)
        return {"aggregate_only": True, "reports": {key: str(value) for key, value in outputs.items()}}

    seeds = parse_seeds(args.seeds)
    gpus = parse_gpus(args.gpus)
    variant_label = getattr(args, "variant_label", None)
    manifest_suffix = f"__{_sanitize_segment(str(args.rq))}"
    if variant_label:
        manifest_suffix = f"{manifest_suffix}__{_sanitize_segment(str(variant_label))}"

    manifest = {
        "suite_name": args.suite_name,
        "rq": args.rq,
        "variant_label": variant_label,
        "datasets": args.datasets,
        "models": args.models,
        "preprocess_paths": {
            dataset: {
                model: str(
                    ensure_processed_data(
                        dataset,
                        model,
                        dry_run=True,
                        rq=args.rq,
                        variant_label=variant_label,
                    )
                )
                for model in args.models
            }
            for dataset in args.datasets
        },
        "preprocess_reuse_policy": _preprocess_reuse_policy(args.rq),
        "seeds": seeds,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "strict_repro": not bool(args.allow_approximate),
        "split_mode": getattr(args, "split_mode", "stratified"),
        "use_temperature_calibration": not bool(getattr(args, "disable_temperature_calibration", False)),
        "parallel": bool(args.parallel),
        "gpus": gpus,
        "dry_run": bool(args.dry_run),
        "require_cuda": not bool(getattr(args, "allow_cpu", False)),
        "enable_internal_probe": bool(getattr(args, "enable_diagnostic_probes", False)),
        "notes": getattr(args, "notes", None),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = base_output_dir / f"suite_manifest{manifest_suffix}.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    _write_variant_index(base_output_dir, args, manifest, seeds, gpus)

    if args.dry_run:
        jobs: List[Dict[str, Any]] = []
        for dataset in args.datasets:
            dataset_info = DATASET_MAP[dataset]
            min_class_samples = getattr(args, "min_class_samples", None)
            if min_class_samples is None:
                min_class_samples = dataset_info.get("min_class_samples", 1)
            for model in args.models:
                processed_data_path = ensure_processed_data(
                    dataset,
                    model,
                    dry_run=True,
                    rq=args.rq,
                    variant_label=variant_label,
                )
                for seed in seeds:
                    run_dir = _build_run_dir(base_output_dir, args.rq, dataset, model, int(seed))
                    if variant_label:
                        run_dir = run_dir.parent / _sanitize_segment(str(variant_label)) / run_dir.name
                    jobs.append(
                        {
                            "rq": args.rq,
                            "suite_name": args.suite_name,
                            "variant_label": variant_label,
                            "dataset": dataset,
                            "model": model,
                            "seed": int(seed),
                            "run_dir": str(run_dir),
                            "gpu_candidates": gpus,
                            "processed_data_path": str(processed_data_path),
                            "preprocess_method": _method_for_model(model),
                            "preprocess_reuse_policy": _preprocess_reuse_policy(args.rq),
                            "config": {
                                "epochs": args.epochs,
                                "lr": args.lr,
                                "batch_size": args.batch_size,
                                "ablation_mode": args.ablation_mode,
                                "use_ctgan": bool(args.use_ctgan),
                                "multi_gpu_dp": bool(args.multi_gpu_dp),
                                "use_compile": bool(args.use_compile),
                                "graph_variant": getattr(args, "graph_variant", "full"),
                                "feature_variant": getattr(args, "feature_variant", "full"),
                                "split_mode": getattr(args, "split_mode", "stratified"),
                                "fusion_mode": getattr(args, "fusion_mode", "adaptive"),
                                "pooling_mode": getattr(args, "pooling_mode", "attention"),
                                "rgcn_num_bases": getattr(args, "rgcn_num_bases", 30),
                                "loss_type": getattr(args, "loss_type", None),
                                "focal_gamma": getattr(args, "focal_gamma", 2.0),
                                "use_temperature_calibration": not bool(getattr(args, "disable_temperature_calibration", False)),
                                "min_class_samples": int(min_class_samples),
                                "strict_repro": not bool(args.allow_approximate),
                                "require_cuda": not bool(getattr(args, "allow_cpu", False)),
                                "enable_internal_probe": bool(getattr(args, "enable_diagnostic_probes", False)),
                            },
                        }
                    )
        plan = {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "n_jobs": len(jobs),
            "jobs": jobs,
        }
        plan_path = base_output_dir / f"suite_plan{manifest_suffix}.json"
        with plan_path.open("w", encoding="utf-8") as fp:
            json.dump(plan, fp, ensure_ascii=False, indent=2)
        return {"dry_run": True, "manifest_path": str(manifest_path), "plan_path": str(plan_path), "n_jobs": len(jobs)}

    group_summaries: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        for model in args.models:
            LOGGER.info("Running group: rq=%s dataset=%s model=%s seeds=%s", args.rq, dataset, model, seeds)
            processed_data_path = ensure_processed_data(
                dataset,
                model,
                dry_run=False,
                rq=args.rq,
                variant_label=variant_label,
            )
            config_template = {
                "suite_name": args.suite_name,
                "rq": args.rq,
                "dataset": dataset,
                "model": model,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "ablation_mode": args.ablation_mode,
                "use_ctgan": bool(args.use_ctgan),
                "base_output_dir": str(base_output_dir),
                "multi_gpu_dp": bool(args.multi_gpu_dp),
                "use_compile": bool(args.use_compile),
                "graph_variant": getattr(args, "graph_variant", "full"),
                "feature_variant": getattr(args, "feature_variant", "full"),
                "split_mode": getattr(args, "split_mode", "stratified"),
                "fusion_mode": getattr(args, "fusion_mode", "adaptive"),
                "pooling_mode": getattr(args, "pooling_mode", "attention"),
                "rgcn_num_bases": getattr(args, "rgcn_num_bases", 30),
                "loss_type": getattr(args, "loss_type", None),
                "focal_gamma": getattr(args, "focal_gamma", 2.0),
                "use_temperature_calibration": not bool(getattr(args, "disable_temperature_calibration", False)),
                "min_class_samples": getattr(args, "min_class_samples", None),
                "strict_repro": not bool(args.allow_approximate),
                "require_cuda": not bool(getattr(args, "allow_cpu", False)),
                "enable_internal_probe": bool(getattr(args, "enable_diagnostic_probes", False)),
                "processed_data_path": str(processed_data_path),
                "preprocess_method": _method_for_model(model),
                "preprocess_reuse_policy": _preprocess_reuse_policy(args.rq),
                "variant_label": variant_label,
                "notes": getattr(args, "notes", None),
            }
            if args.parallel:
                results = run_seeds_parallel(
                    seeds=seeds,
                    worker_fn=suite_worker,
                    config_template=config_template,
                    output_dir=base_output_dir / args.rq / dataset / model,
                    gpus=gpus,
                )
            else:
                gpu_id = -1
                if gpus:
                    gpu_id = int(gpus[0])
                results = run_seeds_sequential(
                    seeds=seeds,
                    worker_fn=suite_worker,
                    config_template=config_template,
                    output_dir=base_output_dir / args.rq / dataset / model,
                    gpu_id=gpu_id,
                )
            failed = [item for item in results if not bool(item.get("ok", False))]
            if failed:
                raise RuntimeError(
                    f"Experiment group failed: rq={args.rq} dataset={dataset} model={model} "
                    f"variant={variant_label} failures={failed}"
                )
            group_summaries.append({"dataset": dataset, "model": model, "results": results})

    reports_dir = base_output_dir / "_reports"
    outputs = aggregate_experiment_tree(base_output_dir, reports_dir)
    summary = {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "groups": group_summaries,
        "reports": {key: str(value) for key, value in outputs.items()},
    }
    with (base_output_dir / f"suite_last_run{manifest_suffix}.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2, default=str)
    return summary


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    summary = run_suite(args)
    print(json.dumps(summary.get("reports", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
