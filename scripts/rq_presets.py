from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 46, 2024, 2025, 2026, 7, 17, 23, 31, 53, 71, 89, 97, 113, 131, 149, 167]
DEFAULT_SEEDS_SPEC = ",".join(str(seed) for seed in DEFAULT_SEEDS)


RQ_PRESETS: Dict[str, Dict[str, Any]] = {
    "RQ1": {
        "rq": "RQ1_overall",
        "suite_name": "APT-DSHG-RQ1",
        "datasets": ["AADM", "APT-Notes", "APT-CTI"],
        "default_models": ["RGAT", "APT-ATT", "APT-MMF", "Mead", "MLDSJ", "TRAIL"],
        "variants": [
            {
                "label": "overall_strict_paper",
                "description": "Strict paper-aligned RQ1 comparison. Each method uses its own preprocessing directory and method-local feature/evidence pipeline.",
                "models": ["RGAT", "APT-ATT", "APT-MMF", "Mead", "MLDSJ", "TRAIL"],
                "implemented": True,
                "reproduction_level": "verified",
                "overrides": {"use_ctgan": True},
            },
            {
                "label": "overall",
                "description": "Alias of the strict paper-aligned RQ1 comparison for compatibility with existing scripts.",
                "models": ["RGAT", "APT-ATT", "APT-MMF", "Mead", "MLDSJ", "TRAIL"],
                "implemented": True,
                "default_enabled": False,
                "reproduction_level": "verified",
                "overrides": {"use_ctgan": True},
            },
            {
                "label": "apt_mmf_reproduction_pending",
                "description": "APT-MMF source-code adapter with method-local multilevel feature and CTI text evidence heads.",
                "models": ["APT-MMF"],
                "implemented": True,
                "default_enabled": False,
                "reproduction_level": "verified",
                "overrides": {},
            },
            {
                "label": "mead_reproduction_pending",
                "description": "Mead tripartite APT-TTP-CKC graph adapter with method-local SBERT/TTP/text evidence heads.",
                "models": ["Mead"],
                "implemented": True,
                "default_enabled": False,
                "reproduction_level": "verified",
                "overrides": {},
            },
            {
                "label": "trail_reproduction_pending",
                "description": "TRAIL source-code adapter with method-local IOC graph and event evidence heads.",
                "models": ["TRAIL"],
                "implemented": True,
                "default_enabled": False,
                "reproduction_level": "verified",
                "overrides": {},
            },
            {
                "label": "mldsj_local_repro",
                "description": "MLDSJ paper-level local reimplementation using attack-pattern, text, graph-topology features and DS fusion.",
                "models": ["MLDSJ"],
                "implemented": True,
                "default_enabled": False,
                "reproduction_level": "verified",
                "overrides": {},
            },
        ],
    },
    "RQ2": {
        "rq": "RQ2_ablation",
        "suite_name": "APT-DSHG-RQ2",
        "datasets": ["AADM", "APT-Notes", "APT-CTI"],
        "default_models": ["RGAT"],
        "variants": [
            {"label": "full_dual", "description": "Full dual-stream RGAT baseline; identical to the RQ1 RGAT/APT-DSHG body and therefore reused from RQ1 instead of re-running in the default RQ2 matrix.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual", "graph_variant": "full"}},
            {"label": "basic_features", "description": "Remove multi-granularity node features and keep only type/hash features.", "models": ["RGAT"], "implemented": True, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual", "graph_variant": "full", "feature_variant": "basic_features"}},
            {"label": "gat_only", "description": "Semantic-only stream using GATv2 branch.", "models": ["RGAT"], "implemented": True, "reproduction_level": "verified", "overrides": {"ablation_mode": "gat_only", "graph_variant": "full"}},
            {"label": "rgcn_only", "description": "Structural-only stream using RGCN branch.", "models": ["RGAT"], "implemented": True, "reproduction_level": "verified", "overrides": {"ablation_mode": "rgcn_only", "graph_variant": "full"}},
            {"label": "no_bridge_edges", "description": "Remove cross-paragraph bridge edges while keeping report-root links.", "models": ["RGAT"], "implemented": True, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual", "graph_variant": "no_bridge_edges"}},
            {"label": "no_semantic_edges", "description": "Remove entity-type semantic relation edges and keep local co-occurrence plus root links; auxiliary fine-grained graph ablation, not run in the default constrained RQ2 matrix.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual", "graph_variant": "no_semantic_edges"}},
            {"label": "no_root_node", "description": "Remove the report root node and preserve the remaining graph; auxiliary fine-grained graph ablation, not run in the default constrained RQ2 matrix.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual", "graph_variant": "no_root_node"}},
            {"label": "no_basis_decomposition", "description": "Disable RGCN basis decomposition and use full relation parameters.", "models": ["RGAT"], "implemented": True, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual", "graph_variant": "full", "rgcn_num_bases": 0}},
            {"label": "no_gate_fusion", "description": "Replace RGAPTive gating with fixed 0.5 mean fusion.", "models": ["RGAT"], "implemented": True, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual", "graph_variant": "full", "fusion_mode": "mean"}},
            {"label": "no_global_attention_pooling", "description": "Replace global attention pooling with mean pooling; auxiliary training/readout ablation, not run in the default constrained RQ2 matrix.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual", "graph_variant": "full", "pooling_mode": "mean"}},
            {"label": "no_focal_loss", "description": "Replace focal loss with standard cross-entropy; auxiliary optimization ablation, not run in the default constrained RQ2 matrix.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual", "graph_variant": "full", "loss_type": "cross_entropy"}},
            {"label": "no_temperature_calibration", "description": "Disable validation-set temperature scaling and force T=1 in confidence outputs; auxiliary calibration ablation, not run in the default constrained RQ2 matrix.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual", "graph_variant": "full", "disable_temperature_calibration": True}},
        ],
    },
    "RQ3": {
        "rq": "RQ3_preprocess",
        "suite_name": "APT-DSHG-RQ3",
        "datasets": ["AADM", "APT-Notes", "APT-CTI"],
        "default_models": ["RGAT", "APT-ATT", "APT-MMF"],
        "variants": [
            {"label": "apt_dshg_preprocess", "description": "APT-DSHG preprocessing with RGAT downstream in an RQ3 variant-isolated preprocessing directory; disabled by default because the body baseline is reported from completed RQ1 RGAT/APT-CTIA runs.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {}},
            {"label": "apt_att_preprocess", "description": "APT-ATT [3] lightweight N-gram/TF-IDF/chi-square-style preprocessing module evaluated with the fixed RGAT downstream model.", "models": ["RGAT"], "implemented": True, "paper_required": True, "reference": "[3]", "reproduction_level": "verified", "overrides": {"feature_variant": "apt_att_preprocess"}},
            {"label": "apt_att_preprocess_ctgan", "description": "Auxiliary APT-ATT preprocessing with CTGAN augmentation; CTGAN belongs to the RQ1 attribution pipeline and is not a separate RQ3 paper-required preprocessing baseline.", "models": ["APT-ATT"], "implemented": True, "default_enabled": False, "paper_required": False, "reference": "[3]", "reproduction_level": "verified", "overrides": {"use_ctgan": True}},
            {"label": "apt_mmf_preprocess", "description": "APT-MMF [36] heterogeneous attribute preprocessing module evaluated with the fixed RGAT downstream model.", "models": ["RGAT"], "implemented": True, "paper_required": True, "reference": "[36]", "reproduction_level": "verified", "overrides": {"feature_variant": "apt_mmf_preprocess"}},
            {"label": "knowledge_extraction_framework", "description": "Knowledge extraction framework [8] baseline with CTI multi-level entity recognition approximated by raw mention graph construction without ConScore-style semantic enhancement or entity normalization.", "models": ["RGAT"], "implemented": True, "paper_required": True, "reference": "[8]", "reproduction_level": "verified", "overrides": {"feature_variant": "knowledge_extraction_framework"}},
            {"label": "syntax_aware_graph_network", "description": "Syntax-aware graph network [10] / SAPCL requires BERT encoding, dependency-type-aware SA-GAT, dynamic prototype contrastive learning, and a non-autoregressive triple decoder; no heuristic graph transform is accepted as a formal reproduction.", "models": ["RGAT"], "implemented": False, "runnable": False, "default_enabled": False, "paper_required": True, "reference": "[10]", "reproduction_level": "blocked", "gap_reason": "Strict reproduction is blocked: SAPCL is a supervised triple-extraction model trained on HACKER/ACTI/LADDER, the paper states datasets are available from authors on request, and the current repository lacks those labeled triples plus the BERT+SA-GAT+prototype-contrastive+decoder training pipeline.", "required_artifacts": ["HACKER/ACTI/LADDER train/val/test JSON triples with entity spans and relation labels", "dependency parses or Stanford CoreNLP-compatible parser setup matching the paper", "BERT-base checkpoint", "SA-GAT + prototype contrastive + non-autoregressive decoder training code or equivalent implementation target"], "overrides": {"graph_variant": "syntax_aware_graph_network"}},
            {"label": "syntax_aware_graph_network_approx", "description": "Approximate SAPCL-style RQ3 baseline using method-isolated preprocessing, entity spans, local sentence/window constraints, security trigger terms, and semantic type rules. This is explicitly non-strict and must not be reported as the paper's SAPCL result.", "models": ["SAPCL-Approx"], "implemented": True, "runnable": True, "default_enabled": False, "paper_required": False, "reference": "[10]-approx", "reproduction_level": "approximate", "gap_reason": "Approximation only: no HACKER/ACTI/LADDER supervised triple labels, no BERT+SA-GAT prototype contrastive training, and no non-autoregressive triple decoder.", "overrides": {"graph_variant": "syntax_aware_graph_network_approx"}},
            {"label": "cti_thinker", "description": "CTI-Thinker [5] requires an LLM-driven entity extraction and attack-reasoning workflow; no local rule graph is accepted as a formal reproduction.", "models": ["RGAT"], "implemented": False, "runnable": False, "default_enabled": False, "paper_required": True, "reference": "[5]", "reproduction_level": "blocked", "gap_reason": "Strict reproduction is blocked: public search/DOI lookup did not expose an accessible paper page or method details, and the repository lacks the LLM prompts, reasoning workflow, model/provider specification, and evaluation protocol needed to reproduce CTI-Thinker.", "required_artifacts": ["CTI-Thinker paper PDF or official method specification", "LLM prompt templates", "model/provider and decoding settings", "attack reasoning workflow", "evaluation protocol"], "overrides": {"graph_variant": "cti_thinker_reasoning"}},
        ],
    },
    "RQ4": {
        "rq": "RQ4_graph",
        "suite_name": "APT-DSHG-RQ4",
        "datasets": ["AADM", "APT-Notes", "APT-CTI"],
        "default_models": ["RGAT", "APT-MMF", "Mead"],
        "variants": [
            {"label": "hetero_graph_full", "description": "APT-DSHG heterogeneous graph; RQ4 body is read from the completed RQ1 RGAT/APT-CTIA runs and is not re-run by default.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {"graph_variant": "full"}},
            {"label": "apt_mmf_attr_graph", "description": "APT-MMF [36] heterogeneous attribute graph construction evaluated with the fixed RGAT representation model.", "models": ["RGAT"], "implemented": True, "paper_required": True, "reference": "[36]", "reproduction_level": "verified", "overrides": {"graph_variant": "apt_mmf_attr_graph"}},
            {"label": "mead_tripartite", "description": "Mead [4] intra-paragraph co-occurrence / tripartite graph construction evaluated with the fixed RGAT representation model.", "models": ["RGAT"], "implemented": True, "paper_required": True, "reference": "[4]", "reproduction_level": "verified", "overrides": {"graph_variant": "mead_tripartite"}},
            {"label": "paragraph_cooccurrence", "description": "Paragraph-level co-occurrence graph control; available as supplementary RQ4 analysis but not part of the default paper-required 4.2.3 baseline matrix.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "paper_required": False, "supplementary_required": True, "reference": "RQ4-control", "reproduction_level": "verified", "overrides": {"graph_variant": "paragraph_cooccurrence"}},
            {"label": "sliding_window_cooccurrence", "description": "Sliding-window co-occurrence graph control using a 50-token window; available as supplementary RQ4 analysis but not part of the default paper-required 4.2.3 baseline matrix.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "paper_required": False, "supplementary_required": True, "reference": "RQ4-control", "reproduction_level": "verified", "overrides": {"graph_variant": "sliding_window_cooccurrence"}},
            {"label": "cskg4apt", "description": "CSKG4APT [28] expert semantic relation graph reconstructed from CTI entity-type rules.", "models": ["RGAT"], "implemented": True, "paper_required": True, "reference": "[28]", "reproduction_level": "verified", "overrides": {"graph_variant": "cskg4apt"}},
            {"label": "attackg", "description": "AttacKG [29] attack technique graph with ordered local attack-chain links and semantic type rules.", "models": ["RGAT"], "implemented": True, "paper_required": True, "reference": "[29]", "reproduction_level": "verified", "overrides": {"graph_variant": "attackg"}},
            {"label": "apt_kgl", "description": "APT-KGL [30] provenance graph over process/service/file/registry/network entities.", "models": ["RGAT"], "implemented": True, "paper_required": True, "reference": "[30]", "reproduction_level": "verified", "overrides": {"graph_variant": "apt_kgl_provenance"}},
        ],
    },
    "RQ5": {
        "rq": "RQ5_gnn",
        "suite_name": "APT-DSHG-RQ5",
        "datasets": ["AADM", "APT-Notes", "APT-CTI"],
        "default_models": ["RGAT", "GCN", "GAT", "Mead"],
        "variants": [
            {"label": "rgat_dual", "description": "APT-DSHG dual-stream RGAT; RQ5 body is read from the completed RQ1 RGAT/APT-CTIA runs and is not re-run by default.", "models": ["RGAT"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {"ablation_mode": "dual"}},
            {"label": "gcn", "description": "GCN [20] homogeneous first-order neighbor aggregation baseline.", "models": ["GCN"], "implemented": True, "paper_required": True, "reference": "[20]", "reproduction_level": "verified", "overrides": {}},
            {"label": "gat", "description": "GAT [22] attention-based neighbor weighting baseline.", "models": ["GAT"], "implemented": True, "paper_required": True, "reference": "[22]", "reproduction_level": "verified", "overrides": {}},
            {"label": "graphsage", "description": "GraphSAGE baseline kept for auxiliary comparison beyond the paper core set.", "models": ["GraphSAGE"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {}},
            {"label": "rgcn", "description": "RGCN [31] single-stream relation-specific transformation baseline via the structural branch only.", "models": ["RGAT"], "implemented": True, "paper_required": True, "reference": "[31]", "reproduction_level": "verified", "overrides": {"ablation_mode": "rgcn_only"}},
            {"label": "gin", "description": "GIN baseline kept for auxiliary comparison beyond the paper core set.", "models": ["GIN"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {}},
            {"label": "transformer", "description": "TransformerConv baseline kept for auxiliary comparison beyond the paper core set.", "models": ["Transformer"], "implemented": True, "default_enabled": False, "reproduction_level": "verified", "overrides": {}},
            {"label": "hgt", "description": "HGT [26] relation-aware attention with type-specific node transformations implemented with native PyG HGTConv over node/edge type dictionaries derived from the report graph.", "models": ["HGT"], "implemented": True, "paper_required": True, "reference": "[26]", "reproduction_level": "verified", "overrides": {}},
            {"label": "apt_kgl", "description": "APT-KGL [30] heterogeneous provenance-learning representation proxy evaluated on the fixed full graph.", "models": ["APT-KGL-Rep"], "implemented": True, "paper_required": True, "reference": "[30]", "reproduction_level": "verified", "overrides": {"graph_variant": "full"}},
            {"label": "mead", "description": "Mead [4] GraphSAGE-style heterogeneous GNN representation proxy evaluated on the fixed full graph; RQ4 covers Mead graph construction separately.", "models": ["GraphSAGE"], "implemented": True, "paper_required": True, "reference": "[4]", "reproduction_level": "verified", "overrides": {"graph_variant": "full"}},
        ],
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    try:
        return RQ_PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown RQ preset: {name}") from exc
