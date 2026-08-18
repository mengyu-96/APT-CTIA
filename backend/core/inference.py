from __future__ import annotations

import io
import json
import logging
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

try:
    from backend.core.repro_paths import ATTACK_JSON_CANDIDATES
except ImportError:
    from core.repro_paths import ATTACK_JSON_CANDIDATES  # type: ignore

try:
    from backend.core.train import (
        APTAttributionGCN,
        APTAttributionGAT,
        APTAttributionGIN,
        APTAttributionGraphSAGE,
        APTAttributionHybrid,
        APTAttributionTransformer,
        load_graph_dataset,
    )
except ImportError:
    try:
        from core.train import (  # type: ignore
            APTAttributionGCN,
            APTAttributionGAT,
            APTAttributionGIN,
            APTAttributionGraphSAGE,
            APTAttributionHybrid,
            APTAttributionTransformer,
            load_graph_dataset,
        )
    except ImportError:
        from train import (  # type: ignore
            APTAttributionGCN,
            APTAttributionGAT,
            APTAttributionGIN,
            APTAttributionGraphSAGE,
            APTAttributionHybrid,
            APTAttributionTransformer,
            load_graph_dataset,
        )

try:
    from backend.core.models.rgat import RelationAwareGAT
except ImportError:
    try:
        from core.models.rgat import RelationAwareGAT  # type: ignore
    except ImportError:
        try:
            from models.rgat import RelationAwareGAT  # type: ignore
        except ImportError:
            RelationAwareGAT = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger(__name__)

if sys.platform == "win32" and getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _score_quality(values: List[float]) -> Dict[str, Any]:
    scores = [max(_safe_float(value), 0.0) for value in values]
    count = len(scores)
    total = sum(scores)
    if count < 2 or total <= 0:
        return {
            "reliable": False,
            "reason": "insufficient_scores",
            "count": count,
            "normalized_entropy": 1.0,
            "coefficient_of_variation": 0.0,
            "top_to_mean": 1.0,
        }

    mean = total / count
    variance = sum((value - mean) ** 2 for value in scores) / count
    coefficient_of_variation = math.sqrt(variance) / mean if mean > 0 else 0.0
    probabilities = [value / total for value in scores]
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0)
    normalized_entropy = entropy / math.log(count)
    top_to_mean = max(scores) / mean
    reliable = (
        coefficient_of_variation >= 0.05
        and top_to_mean >= 1.10
        and normalized_entropy <= 0.98
    )
    return {
        "reliable": reliable,
        "reason": "differentiated" if reliable else "near_uniform_distribution",
        "count": count,
        "normalized_entropy": float(normalized_entropy),
        "coefficient_of_variation": float(coefficient_of_variation),
        "top_to_mean": float(top_to_mean),
    }


def _load_attack_mapping() -> Dict[str, List[str]]:
    for path in ATTACK_JSON_CANDIDATES:
        if not path.exists() or path.stat().st_size <= 0:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fp:
                payload = json.load(fp)
            mapping: Dict[str, List[str]] = {}
            for obj in payload.get("objects", []):
                if obj.get("type") != "attack-pattern":
                    continue
                technique_id = ""
                for ref in obj.get("external_references", []):
                    ext_id = str(ref.get("external_id", "")).upper()
                    if ref.get("source_name") == "mitre-attack" and ext_id.startswith("T"):
                        technique_id = ext_id.split(".")[0]
                        break
                if not technique_id:
                    continue
                tactics = []
                for phase in obj.get("kill_chain_phases", []):
                    if phase.get("kill_chain_name") != "mitre-attack":
                        continue
                    phase_name = str(phase.get("phase_name", "")).replace("-", " ").strip().lower()
                    if phase_name:
                        tactics.append(phase_name)
                if tactics:
                    mapping[technique_id] = sorted(set(tactics))
            LOGGER.info("Loaded ATT&CK mapping for %d techniques", len(mapping))
            return mapping
        except Exception as exc:
            LOGGER.warning("Failed to parse ATT&CK mapping from %s: %s", path, exc)
    # Offline fallback for the techniques emitted by the built-in behavior
    # extractor.  External STIX data, when present, still takes precedence.
    mapping = {
        "T1005": ["collection"],
        "T1027": ["defense evasion"],
        "T1041": ["exfiltration"],
        "T1059": ["execution"],
        "T1082": ["discovery"],
        "T1083": ["discovery"],
        "T1105": ["command and control"],
        "T1140": ["defense evasion"],
        "T1195": ["initial access"],
        "T1622": ["defense evasion"],
    }
    LOGGER.info("Using built-in ATT&CK tactic mapping for %d extracted techniques", len(mapping))
    return mapping


def _load_entities_by_report(dataset_dir: Path) -> Dict[str, Dict[int, List[dict]]]:
    entities_path = dataset_dir / "entities.jsonl"
    grouped: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    if not entities_path.exists():
        return {}
    with entities_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                record = json.loads(line)
            except Exception:
                continue
            report_id = record.get("report_id")
            if not report_id:
                continue
            para_idx = int(record.get("paragraph_index", 0))
            grouped[report_id][para_idx].append(record)
    return {rid: dict(paras) for rid, paras in grouped.items()}


def _extract_state_dict(blob: Any) -> Any:
    if isinstance(blob, dict):
        for key in ["state_dict", "model_state_dict", "weights"]:
            if key in blob and isinstance(blob[key], dict):
                return blob[key]
    return blob


def _compat_remap_rgat_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        new_key = key
        if key.startswith("conv1."):
            new_key = "gat_conv." + key[len("conv1."):]
        elif key.startswith("rgcn_branch."):
            new_key = "rgcn_conv." + key[len("rgcn_branch."):]
        elif key.startswith("fusion_gate."):
            continue
        remapped[new_key] = value
    return remapped


def _filter_compatible_keys(sd: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    model_sd = model.state_dict()
    kept: Dict[str, torch.Tensor] = {}
    dropped: List[Tuple[str, Tuple[int, ...], Optional[Tuple[int, ...]]]] = []
    for key, value in sd.items():
        if key in model_sd and tuple(value.shape) == tuple(model_sd[key].shape):
            kept[key] = value
        else:
            expected = tuple(model_sd[key].shape) if key in model_sd else None
            dropped.append((key, tuple(value.shape), expected))
    if dropped:
        preview = ", ".join(f"{k}: {src}->{dst}" for k, src, dst in dropped[:10])
        LOGGER.warning(
            "Dropped %d incompatible weight(s): %s%s",
            len(dropped),
            preview,
            " ..." if len(dropped) > 10 else "",
        )
    return kept


def _load_feature_manifest(path: Path) -> Optional[Dict[str, Any]]:
    manifest_path = path / "feature_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as fp:
            manifest = json.load(fp)
    except Exception as exc:
        raise RuntimeError(f"Failed to read feature manifest at {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or not manifest.get("fingerprint"):
        raise RuntimeError(f"Invalid feature manifest at {manifest_path}")
    return manifest


def _assert_feature_compatibility(
    model_dir: Path,
    dataset_dir: Path,
    train_config: Dict[str, Any],
) -> None:
    model_manifest = _load_feature_manifest(model_dir)
    if model_manifest is None:
        training_data_path = train_config.get("processed_data_path")
        if training_data_path:
            model_manifest = _load_feature_manifest(Path(training_data_path))
    dataset_manifest = _load_feature_manifest(dataset_dir)

    if model_manifest is None and dataset_manifest is None:
        LOGGER.warning("Feature manifests are missing; treating model and dataset as legacy artifacts.")
        return
    if model_manifest is None or dataset_manifest is None:
        raise RuntimeError(
            "Feature contract mismatch: one artifact is legacy and the other is versioned. "
            "Reprocess the training data and retrain the model before inference."
        )
    if model_manifest["fingerprint"] != dataset_manifest["fingerprint"]:
        raise RuntimeError(
            "Feature contract mismatch between model and inference dataset: "
            f"model={model_manifest['fingerprint'][:12]}, "
            f"dataset={dataset_manifest['fingerprint'][:12]}. "
            "Use a model trained from the same preprocessing feature schema."
        )


def load_model_for_inference(
    model_path: Path,
    config: Dict[str, Any],
    num_classes: int,
    input_dim: int,
    text_emb_dim: int = 0,
):
    model_type = config.get("model_type", "GAT")
    hidden_dim = int(config.get("hidden_dim", 128))
    dropout = float(config.get("dropout", 0.5))
    heads = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "GCN":
        model = APTAttributionGCN(input_dim, hidden_dim, num_classes, dropout=dropout)
    elif model_type == "GAT":
        model = APTAttributionGAT(input_dim, hidden_dim, num_classes, heads=heads, dropout=dropout)
    elif model_type == "Transformer":
        model = APTAttributionTransformer(input_dim, hidden_dim, num_classes, heads=heads, dropout=dropout)
    elif model_type == "GraphSAGE":
        model = APTAttributionGraphSAGE(input_dim, hidden_dim, num_classes, dropout=dropout, text_emb_dim=text_emb_dim)
    elif model_type == "GIN":
        model = APTAttributionGIN(input_dim, hidden_dim, num_classes, dropout=dropout)
    elif model_type == "RGAT":
        if RelationAwareGAT is None:
            raise RuntimeError("RGAT model code not available.")
        model = RelationAwareGAT(
            num_node_features=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            ablation_mode=config.get("ablation_mode", "dual"),
        )
    else:
        model = APTAttributionHybrid(input_dim, hidden_dim, num_classes, model_type=model_type, heads=heads, dropout=dropout)

    try:
        raw_blob = torch.load(model_path, map_location=device)
        state_dict = _extract_state_dict(raw_blob)
        if model_type == "RGAT":
            state_dict = _compat_remap_rgat_keys(state_dict)
        state_dict = _filter_compatible_keys(state_dict, model)
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing or unexpected:
            LOGGER.warning("Loaded model with non-strict keys. Missing: %s | Unexpected: %s", missing, unexpected)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model weights from {model_path}: {exc}") from exc

    model = model.to(device)
    model.eval()
    return model, device


def _graph_edges_from_data(graph: Data) -> List[Tuple[int, int]]:
    if graph.edge_index is None or graph.edge_index.numel() == 0:
        return []
    edge_index = graph.edge_index.detach().cpu()
    return [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.size(1))]


def _build_node_metadata(graph: Data, report_entities: Dict[int, List[dict]]) -> List[dict]:
    node_count = int(graph.x.size(0))
    node_texts = list(getattr(graph, "node_texts", []) or [])
    node_labels = list(getattr(graph, "node_labels", []) or [])
    node_paragraphs = list(getattr(graph, "node_paragraph_indices", []) or [])
    if len(node_texts) == node_count and len(node_labels) == node_count:
        entity_lookup: Dict[Tuple[str, str], dict] = {}
        for paragraph_entries in report_entities.values():
            for entry in paragraph_entries:
                key = (str(entry.get("text", "")).strip().lower(), str(entry.get("label", "UNKNOWN")))
                if key[0] and key not in entity_lookup:
                    entity_lookup[key] = entry
        nodes = []
        for idx in range(node_count):
            text = str(node_texts[idx])
            label = str(node_labels[idx])
            entry = entity_lookup.get((text.strip().lower(), label), {})
            nodes.append({
                "id": idx,
                "text": text,
                "label": label,
                "paragraph_index": int(node_paragraphs[idx]) if idx < len(node_paragraphs) else -1,
                "source": entry.get("source"),
                "mapping_reason": entry.get("mapping_reason"),
                "start": entry.get("start"),
                "end": entry.get("end"),
            })
        return nodes

    nodes = [{
        "id": 0,
        "text": str(getattr(graph, "report_id", "REPORT")),
        "label": "REPORT",
        "paragraph_index": -1,
    }]
    seen = set()
    for para_idx in sorted(report_entities.keys()):
        for entry in report_entities.get(para_idx, []):
            text = str(entry.get("text", "")).strip()
            label = str(entry.get("label", "UNKNOWN"))
            key = (text.lower(), label)
            if not text or key in seen:
                continue
            seen.add(key)
            nodes.append({
                "id": len(nodes),
                "text": text,
                "label": label,
                "paragraph_index": int(para_idx),
            })
    while len(nodes) < node_count:
        idx = len(nodes)
        nodes.append({
            "id": idx,
            "text": f"NODE_{idx}",
            "label": "UNKNOWN",
            "paragraph_index": -1,
        })
    return nodes[:node_count]


def _edge_importance_map(attention_data: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
    edge_att = attention_data.get("edge_attention") or []
    edge_index = attention_data.get("edge_index") or []
    if len(edge_index) != 2:
        return {}
    weights: Dict[Tuple[int, int], float] = {}
    for i, weight in enumerate(edge_att):
        if i >= len(edge_index[0]) or i >= len(edge_index[1]):
            break
        source, target = int(edge_index[0][i]), int(edge_index[1][i])
        if source == target:
            continue
        pair = tuple(sorted((source, target)))
        weights[pair] = max(weights.get(pair, 0.0), _safe_float(weight))
    return weights


def _shortest_path(start: int, goal: int, edges: List[Tuple[int, int]]) -> List[int]:
    if start == goal:
        return [start]
    adjacency: Dict[int, set[int]] = defaultdict(set)
    for src, dst in edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for nxt in adjacency.get(node, set()):
            if nxt in visited:
                continue
            new_path = path + [nxt]
            if nxt == goal:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return []


def _edge_relation_map(graph: Data) -> Dict[Tuple[int, int], str]:
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None:
        return {}
    relation_names = list(getattr(graph, "edge_relation_names", []) or [])
    relation_ids = getattr(graph, "edge_type", None)
    id_to_name = {
        0: "local_cooccurrence", 1: "report_anchor", 2: "self_evidence",
        3: "behavior_evidence", 4: "malware_artifact",
        5: "operation_campaign", 6: "entity_context",
    }
    result: Dict[Tuple[int, int], str] = {}
    for idx, (source, target) in enumerate(edge_index.t().tolist()):
        if source == target:
            continue
        if idx < len(relation_names):
            name = str(relation_names[idx])
        elif isinstance(relation_ids, torch.Tensor) and idx < relation_ids.numel():
            name = id_to_name.get(int(relation_ids[idx]), "entity_context")
        else:
            name = "entity_context"
        result[tuple(sorted((int(source), int(target))))] = name
    return result


def _summarize_gate(attention_data: Dict[str, Any]) -> Dict[str, Any]:
    semantic = _safe_float(attention_data.get("semantic_gate_mean"), 0.5)
    structural = _safe_float(attention_data.get("structural_gate_mean"), 1.0 - semantic)
    if semantic >= 0.6:
        dominant = "semantic"
        desc = "Model preference leans toward semantic evidence."
    elif semantic <= 0.4:
        dominant = "structural"
        desc = "Model preference leans toward structural evidence."
    else:
        dominant = "balanced"
        desc = "Model balances semantic and structural evidence."
    return {
        "semantic_gate": semantic,
        "structural_gate": structural,
        "dominant_signal": dominant,
        "description": desc,
    }


def _build_mitre_summary(nodes: List[dict], attention_data: Dict[str, Any], attack_map: Dict[str, List[str]]) -> Dict[str, Any]:
    node_attention = attention_data.get("node_attention") or []
    techniques = []
    tactic_scores: Dict[str, float] = defaultdict(float)
    seen = set()
    for idx, node in enumerate(nodes):
        if node.get("label") != "MITRE_TECH":
            continue
        raw_tid = str(node.get("text", "")).upper().strip()
        if not raw_tid.startswith("T"):
            continue
        technique_id = raw_tid.split(".")[0]
        if technique_id in seen:
            continue
        seen.add(technique_id)
        score = _safe_float(node_attention[idx] if idx < len(node_attention) else 0.0)
        tactics = attack_map.get(technique_id, [])
        techniques.append({
            "technique_id": technique_id,
            "score": score,
            "tactics": tactics,
        })
        for tactic in tactics:
            tactic_scores[tactic] += score if score > 0 else 1.0
    techniques.sort(key=lambda item: item["score"], reverse=True)
    return {
        "techniques": techniques[:8],
        "top_tactics": [
            {"name": name, "score": score}
            for name, score in sorted(tactic_scores.items(), key=lambda item: item[1], reverse=True)[:5]
        ],
    }


def _build_graph_payload(nodes: List[dict], edges: List[Tuple[int, int]], attention_data: Dict[str, Any]) -> Dict[str, Any]:
    node_attention = attention_data.get("node_attention") or []
    edge_weights = _edge_importance_map(attention_data)
    graph_nodes = []
    for idx, node in enumerate(nodes):
        graph_nodes.append({
            "id": node["id"],
            "label": node["text"],
            "type": node["label"],
            "importance": _safe_float(node_attention[idx] if idx < len(node_attention) else 0.0),
            "paragraph_index": node.get("paragraph_index", -1),
        })
    graph_edges = []
    seen = set()
    for src, dst in edges:
        if src == dst:
            continue
        pair = tuple(sorted((src, dst)))
        if pair in seen:
            continue
        seen.add(pair)
        graph_edges.append({
            "source": pair[0],
            "target": pair[1],
            "weight": edge_weights.get(pair, 0.0),
        })
    graph_edges.sort(key=lambda item: item["weight"], reverse=True)
    return {"nodes": graph_nodes, "edges": graph_edges}


def _build_explanation(
    graph: Data,
    attention_data: Dict[str, Any],
    report_entities: Dict[int, List[dict]],
    attack_map: Dict[str, List[str]],
) -> Dict[str, Any]:
    nodes = _build_node_metadata(graph, report_entities)
    edges = _graph_edges_from_data(graph)
    node_attention = attention_data.get("node_attention") or []
    graph_data = _build_graph_payload(nodes, edges, attention_data)

    evidence_node_scores = [
        _safe_float(node_attention[idx] if idx < len(node_attention) else 0.0)
        for idx, node in enumerate(nodes)
        if not (idx == 0 and node.get("label") == "REPORT")
    ]
    node_quality = _score_quality(evidence_node_scores)
    edge_importance = _edge_importance_map(attention_data)
    edge_quality = _score_quality(list(edge_importance.values()))
    relation_map = _edge_relation_map(graph)
    evidence_quality = {
        "reliable": bool(node_quality["reliable"] or edge_quality["reliable"]),
        "node_attention": node_quality,
        "edge_attention": edge_quality,
    }

    key_nodes = []
    if node_quality["reliable"]:
        for idx, node in enumerate(nodes):
            if (idx == 0 and node.get("label") == "REPORT") or node.get("label") == "DIRECT_ACTOR_MENTION":
                continue
            key_nodes.append({
                "node_id": node["id"],
                "text": node["text"],
                "type": node["label"],
                "attention": _safe_float(node_attention[idx] if idx < len(node_attention) else 0.0),
                "evidence_score": _safe_float(node_attention[idx] if idx < len(node_attention) else 0.0),
                "evidence_basis": "node_attention",
                "paragraph_index": node.get("paragraph_index", -1),
                "source": node.get("source"),
                "mapping_reason": node.get("mapping_reason"),
            })
        key_nodes.sort(key=lambda item: item["evidence_score"], reverse=True)
        key_nodes = key_nodes[:8]
    elif edge_quality["reliable"]:
        # Edge attention remains discriminative even when pooling attention is
        # flat. Derive node evidence only from non-root, non-self relation
        # endpoints and label it explicitly as edge-supported evidence.
        incident_support: Dict[int, float] = defaultdict(float)
        for (src, dst), weight in edge_importance.items():
            if src == dst or src == 0 or dst == 0:
                continue
            incident_support[src] = max(incident_support[src], _safe_float(weight))
            incident_support[dst] = max(incident_support[dst], _safe_float(weight))
        for idx, score in sorted(incident_support.items(), key=lambda item: item[1], reverse=True):
            if idx >= len(nodes) or nodes[idx].get("label") in {"REPORT", "DIRECT_ACTOR_MENTION"}:
                continue
            node = nodes[idx]
            key_nodes.append({
                "node_id": node["id"],
                "text": node["text"],
                "type": node["label"],
                "attention": _safe_float(node_attention[idx] if idx < len(node_attention) else 0.0),
                "evidence_score": _safe_float(score),
                "evidence_basis": "incident_edge_attention",
                "paragraph_index": node.get("paragraph_index", -1),
                "source": node.get("source"),
                "mapping_reason": node.get("mapping_reason"),
            })
            if len(key_nodes) >= 8:
                break

    evidence_quality["key_node_basis"] = (
        "node_attention" if node_quality["reliable"]
        else "incident_edge_attention" if key_nodes
        else "none"
    )
    evidence_quality["key_node_count"] = len(key_nodes)

    key_node_by_id = {int(item["node_id"]): item for item in key_nodes}
    for graph_node in graph_data.get("nodes", []):
        key_node = key_node_by_id.get(int(graph_node.get("id", -1)))
        graph_node["raw_attention"] = graph_node.get("importance", 0.0)
        graph_node["is_key_evidence"] = key_node is not None
        if key_node is not None:
            graph_node["importance"] = key_node["evidence_score"]
            graph_node["evidence_basis"] = key_node["evidence_basis"]
        else:
            graph_node["evidence_basis"] = "node_attention"

    key_edges = []
    if edge_quality["reliable"]:
        for (src, dst), weight in sorted(edge_importance.items(), key=lambda item: item[1], reverse=True)[:8]:
            key_edges.append({
                "source": src,
                "target": dst,
                "source_text": nodes[src]["text"] if src < len(nodes) else f"NODE_{src}",
                "target_text": nodes[dst]["text"] if dst < len(nodes) else f"NODE_{dst}",
                "source_type": nodes[src]["label"] if src < len(nodes) else "UNKNOWN",
                "target_type": nodes[dst]["label"] if dst < len(nodes) else "UNKNOWN",
                "weight": _safe_float(weight),
                "relation": relation_map.get((src, dst), "entity_context"),
            })

    evidence_paths = []
    root_id = 0 if nodes and nodes[0].get("label") == "REPORT" else None
    if root_id is not None:
        for node in key_nodes[:5]:
            path = _shortest_path(root_id, int(node["node_id"]), edges)
            if not path:
                continue
            evidence_paths.append({
                "target_node_id": node["node_id"],
                "target_text": node["text"],
                "score": node["evidence_score"],
                "evidence_basis": node["evidence_basis"],
                "path_node_ids": path,
                "path_texts": [nodes[path_idx]["text"] for path_idx in path if path_idx < len(nodes)],
            })

    gate_summary = _summarize_gate(attention_data)
    mitre_summary = _build_mitre_summary(nodes, attention_data, attack_map)
    summary_lines = []
    if key_nodes:
        summary_lines.append(
            "Top evidence nodes: " + ", ".join(
                f"{item['text']} ({item['type']}, {item['evidence_score']:.3f})" for item in key_nodes[:3]
            )
        )
    if not node_quality["reliable"]:
        if key_nodes:
            summary_lines.append(
                "Node attention is nearly uniform; displayed nodes are supported by differentiated non-self edge attention."
            )
        else:
            summary_lines.append(
                "Node attention is nearly uniform and no differentiated relation evidence is available; key nodes are withheld."
            )
    if not evidence_quality["reliable"]:
        summary_lines.append(
            "Attention scores are nearly uniform; key evidence is withheld because differentiation is insufficient."
        )
    if mitre_summary["techniques"]:
        summary_lines.append(
            "Mapped ATT&CK techniques: " + ", ".join(item["technique_id"] for item in mitre_summary["techniques"][:5])
        )
    summary_lines.append(gate_summary["description"])

    return {
        "decision_mode": gate_summary,
        "evidence_quality": evidence_quality,
        "key_nodes": key_nodes,
        "key_edges": key_edges,
        "evidence_paths": evidence_paths,
        "mitre_attack": mitre_summary,
        "graph_data": graph_data,
        "summary_lines": summary_lines,
    }


def _extract_batch_attention(batch: Data, att_data: Dict[str, Any], graph_idx: int) -> Dict[str, Any]:
    node_mask = batch.batch == graph_idx
    node_attention_full = att_data.get("node_attention")
    edge_index_full = att_data.get("edge_index")
    edge_attention_full = att_data.get("edge_attention")
    gate_alpha_full = att_data.get("gate_alpha")

    if node_attention_full is not None:
        node_attention = node_attention_full[node_mask].detach().cpu().numpy().flatten().tolist()
    else:
        node_attention = []

    if edge_index_full is not None and edge_attention_full is not None:
        src_nodes = edge_index_full[0]
        edge_graph_indices = batch.batch[src_nodes]
        edge_mask = edge_graph_indices == graph_idx
        min_node_idx = int(torch.where(node_mask)[0].min().item())
        edge_index = (edge_index_full[:, edge_mask] - min_node_idx).detach().cpu().numpy().tolist()
        edge_attention = edge_attention_full[edge_mask].mean(dim=1).detach().cpu().numpy().tolist()
    else:
        edge_index = []
        edge_attention = []

    if gate_alpha_full is not None:
        gate_alpha = gate_alpha_full[node_mask].detach().cpu().numpy().flatten().tolist()
        semantic_gate_mean = float(sum(gate_alpha) / max(len(gate_alpha), 1))
    else:
        gate_alpha = []
        semantic_gate_mean = _safe_float(att_data.get("semantic_gate_mean"), 0.5)
    structural_gate_mean = 1.0 - semantic_gate_mean

    return {
        "node_attention": node_attention,
        "edge_attention": edge_attention,
        "edge_index": edge_index,
        "gate_alpha": gate_alpha,
        "semantic_gate_mean": semantic_gate_mean,
        "structural_gate_mean": structural_gate_mean,
    }


def run_inference_pipeline(
    model_dir: str | Path,
    dataset_dir: str | Path,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)

    results_path = model_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Model config not found at {results_path}")
    with results_path.open("r", encoding="utf-8") as fp:
        train_results = json.load(fp)
    config = train_results.get("config", {})
    _assert_feature_compatibility(model_dir, dataset_dir, config)

    class_names = []
    if "classification_report" in train_results:
        keys = list(train_results["classification_report"].keys())
        class_names = [k for k in keys if k not in ["accuracy", "macro avg", "weighted avg"]]
        class_names.sort()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}")
    try:
        graphs, _, _ = load_graph_dataset(dataset_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to load graphs from {dataset_dir}: {exc}") from exc
    if not graphs:
        raise ValueError("No valid graphs found in dataset.")

    input_dim = graphs[0].x.size(1)
    text_emb_dim = graphs[0].doc_emb.size(1) if hasattr(graphs[0], "doc_emb") and graphs[0].doc_emb is not None else 0

    train_data_path = config.get("processed_data_path")
    if not class_names and train_data_path:
        train_data_path = Path(train_data_path)
        train_map_path = train_data_path / "label_mapping.json"
        if not train_map_path.exists():
            train_map_path = train_data_path.parent / "label_mapping.json"
        if train_map_path.exists():
            with train_map_path.open("r", encoding="utf-8") as fp:
                mapping = json.load(fp)
            class_names = [item[0] for item in sorted(mapping.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    if num_classes == 0:
        raise RuntimeError("Could not determine number of classes from model config or results.")

    model_path = model_dir / "best_model.pt"
    model, device = load_model_for_inference(model_path, config, num_classes, input_dim, text_emb_dim)

    entities_by_report = _load_entities_by_report(dataset_dir)
    attack_map = _load_attack_mapping()

    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    all_preds = []
    all_probs = []
    all_attentions = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            doc_emb = batch.doc_emb if hasattr(batch, "doc_emb") else None
            att_data = None

            if isinstance(model, APTAttributionGraphSAGE):
                out = model(batch.x, batch.edge_index, batch.batch, doc_emb=doc_emb)
            elif RelationAwareGAT is not None and isinstance(model, RelationAwareGAT):
                out, att_data = model(batch.x, batch.edge_index, batch.batch, getattr(batch, "edge_type", None), return_attention=True)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)

            probs = F.softmax(out, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            if att_data:
                for graph_idx in range(batch.num_graphs):
                    all_attentions.append(_extract_batch_attention(batch, att_data, graph_idx))
            else:
                all_attentions.extend([{}] * batch.num_graphs)

    inference_results = []
    for idx, graph in enumerate(graphs):
        report_id = getattr(graph, "report_id", f"unknown_{idx}")
        pred_idx = int(all_preds[idx])
        prob_vec = all_probs[idx]
        top3_indices = prob_vec.argsort()[-3:][::-1]
        attention_data = all_attentions[idx] if idx < len(all_attentions) else {}
        explanation = _build_explanation(graph, attention_data, entities_by_report.get(report_id, {}), attack_map)
        inference_results.append({
            "report_id": report_id,
            "predicted_label": class_names[pred_idx],
            "confidence": float(prob_vec[pred_idx]),
            "top3": [
                {
                    "label": class_names[j],
                    "confidence": float(prob_vec[j]),
                    "score": float(prob_vec[j]),
                }
                for j in top3_indices
            ],
            "attention_data": attention_data,
            "graph_data": explanation["graph_data"],
            "explanation": explanation,
        })

    label_counts: Dict[str, int] = {}
    for item in inference_results:
        label = item["predicted_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    summary = {
        "total_samples": len(inference_results),
        "label_distribution": label_counts,
        "results": inference_results,
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary["output_dir"] = str(output_path)
        with (output_path / "inference_results.json").open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)

    return summary
