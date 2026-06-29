# APT-CTIA Frontend/Backend Release

This branch contains the runnable frontend and backend application only.

It intentionally excludes experiment scripts, reproduction code, datasets, generated results, model checkpoints, cache files, and IDE metadata.

## Contents

- `backend/`: Flask API, preprocessing, training, inference, report generation, model definitions, and APT group metadata.
- `frontend/`: Streamlit UI and static assets.
- `utils/timing.py`, `utils/splits.py`: runtime utilities required by backend training.
- `docker-compose.yml`: local two-service startup template.

## Run With Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

The default compose setup is now tuned for a demo deployment:

- training submission is disabled by default
- preprocessing, inference, report export, and model/result browsing stay enabled
- frontend auth credentials come from `.env`
- backend runs with `gunicorn` instead of the Flask dev server

Services:

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:5001`

Runtime data directories are mounted but not versioned:

- `dataset_TXT/`
- `uploads/`
- `results_archive/`

## Demo Deployment Strategy

For a low-cost competition demo, the recommended approach is:

1. train models locally
2. copy only the required runtime artifacts to the server
3. keep `ENABLE_TRAINING=false` on the server

Artifacts typically needed on the server:

- raw reports under `dataset_TXT/` if you want live preprocessing
- processed datasets under `results_archive/processed_data/`
- trained models under `results_archive/training_runs/`
- optional past attribution outputs under `results_archive/attribution_results/`

This keeps the server focused on display, preprocessing, inference, and report generation instead of expensive training workloads.

Detailed deployment notes are in `DEPLOYMENT.md`.

## Run Locally

Backend:

```bash
python -m venv .venv-backend
source .venv-backend/bin/activate
pip install -r backend/requirements.txt
PYTHONPATH=backend:. python backend/api.py
```

Frontend:

```bash
python -m venv .venv-frontend
source .venv-frontend/bin/activate
pip install -r frontend/requirements.txt
cd frontend
streamlit run app.py
```

## Notes

- Large local embedding/model files are not included. If offline preprocessing is required, place compatible model assets under `all-MiniLM-L6-v2/` locally.
- Generated datasets, training runs, reports, and attribution results are written under ignored runtime directories.
