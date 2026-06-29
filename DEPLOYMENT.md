# Deployment Guide

## Goal

This project can be deployed as a competition demo service without exposing model training. The recommended server role is:

- keep preprocessing enabled
- keep attribution inference enabled
- keep report export enabled
- disable training submission and hide the training UI

## Recommended Server Size

For a demo-only deployment with training disabled:

- minimum: `2 vCPU / 4 GB RAM / 40 GB SSD`
- safer choice: `4 vCPU / 8 GB RAM / 60 GB SSD`

Choose the safer tier if you will:

- preprocess PDFs on demand
- keep several trained models on disk
- run multiple demos concurrently

If you enable training on the server, this sizing guidance no longer applies. Training should stay on a local workstation or a separate GPU box.

## Recommended Artifact Strategy

To keep server cost down, prepare heavy artifacts locally and upload only what the demo needs:

- `results_archive/processed_data/`
- `results_archive/training_runs/`
- optional `results_archive/attribution_results/`
- optional `dataset_TXT/` if you want live raw-file preprocessing

This avoids spending server CPU time on repeated preprocessing and avoids any training workload during the demo.

## Environment Variables

Start from `.env.example`:

```bash
cp .env.example .env
```

Important flags:

- `ENABLE_TRAINING=false`
- `SHOW_TRAINING_UI=false`
- `ALLOW_TRAINING_REPORT_SOURCE=false`
- `ENABLE_PREPROCESSING=true`
- `ENABLE_INFERENCE=true`
- `ENABLE_UI_AUTH=true`

Set a real password before deployment:

```env
AUTH_USERNAME=demo
AUTH_PASSWORD=replace-with-a-strong-password
AUTH_SESSION_SECRET=replace-with-a-random-secret
```

## Docker Deployment

```bash
docker compose up -d --build
```

Default ports:

- frontend: `8501`
- backend: `5001`

## Suggested Demo Workflow

1. Train and validate models locally.
2. Copy the selected processed dataset and trained model directories to the server.
3. Keep training disabled in `.env`.
4. Use the deployed frontend for preprocessing, attribution, result browsing, and report export.

## Notes

- The repository does not include large datasets, checkpoints, or local embedding assets.
- If preprocessing depends on a local sentence-transformer model, place the compatible files under `all-MiniLM-L6-v2/` on the server manually.
- The current compose file mounts runtime directories from the project root, so keep backups of `results_archive/` before the demo.
