# APT-MMF and TRAIL RQ1 Reproduction

This branch contains the runnable adapters used by the formal RQ1 experiments for `APT-MMF` and `TRAIL`.

The implementation is not a standalone copy of only `APT归因复现/`. Current runs use:

- original reproduction source under `APT归因复现/`
- backend adapters under `backend/core/baselines/`
- unified experiment entrypoints under `scripts/`
- shared split, timing, and runner utilities under `utils/`

Large local corpora, generated experiment artifacts, cached embeddings, and model checkpoints are intentionally excluded from Git.

## Included Source

APT-MMF:

- `backend/core/baselines/apt_mmf.py`
- `APT归因复现/基于多模态与多级特征融合的高级持续性威胁行为者归因方法/APT-MMF/Main.py`
- `APT归因复现/基于多模态与多级特征融合的高级持续性威胁行为者归因方法/APT-MMF/data_loader.py`
- `APT归因复现/基于多模态与多级特征融合的高级持续性威胁行为者归因方法/APT-MMF/model.py`
- `APT归因复现/基于多模态与多级特征融合的高级持续性威胁行为者归因方法/APT-MMF/model_utils.py`

TRAIL:

- `backend/core/baselines/trail.py`
- `APT归因复现/Trail/process_local_reports.py`
- `APT归因复现/Trail/build_local_graph.py`
- `APT归因复现/Trail/Trail-main/src/train_gnn.py`
- `APT归因复现/Trail/Trail-main/src/models/gnn.py`
- `APT归因复现/Trail/Trail-main/src/config.py`
- supporting source under `APT归因复现/Trail/Trail-main/src/`

## Local Inputs Required

Place formal datasets under:

```text
dataset_TXT/AADM
dataset_TXT/APT-Notes
dataset_TXT/apt_groups
```

The unified preprocessing step writes method-local metadata under:

```text
results_archive/processed_data/APT-MMF/<dataset>
results_archive/processed_data/TRAIL/<dataset>
```

These paths are not committed because they contain generated data and local corpus metadata.

For APT-MMF, `enterprise-attack.json` should be available at one of:

```text
APT归因复现/基于多模态与多级特征融合的高级持续性威胁行为者归因方法/external_knowledge/attack/enterprise-attack.json
APT归因复现/基于多模态与多级特征融合的高级持续性威胁行为者归因方法/external_knowledge/cache/enterprise-attack.json
```

The BERT cache/model is intentionally not committed. Put it under the original expected HuggingFace cache location or:

```text
APT归因复现/基于多模态与多级特征融合的高级持续性威胁行为者归因方法/external_knowledge/hf_models/bert-base-cased
```

## Commands

Dry run:

```bash
python3 scripts/run_RQ1.py --variant overall_strict_paper --models APT-MMF TRAIL --datasets AADM APT-Notes APT-CTI
```

Execute:

```bash
conda run -n APT-CTIA python scripts/run_RQ1.py --execute --variant overall_strict_paper --models APT-MMF TRAIL --datasets AADM APT-Notes APT-CTI --gpus 0 --batch-size 4
```

Formal execution requires CUDA unless `--allow-cpu` is explicitly used for debugging.

## Current Seed-42 Snapshot

A lightweight snapshot is committed at:

```text
docs/generated/apt_mmf_trail_seed42_metrics_snapshot.json
```

It was generated from local `results_archive` metrics and records only the core seven metrics for `APT-MMF` and `TRAIL` on RQ1 `overall_strict_paper`.
