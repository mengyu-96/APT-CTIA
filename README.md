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
docker compose up --build
```

Services:

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:5001`

Runtime data directories are mounted but not versioned:

- `dataset_TXT/`
- `uploads/`
- `reports/`
- `results_archive/`

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
