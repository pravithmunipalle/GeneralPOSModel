# TrialFusion Light

Train a lightweight TrialBench outcome model and serve a front-facing clinical-trial prediction screen.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If TrialBench requires the MeSH embedding file, download `mesh_embeddings.txt.gz` from the ML2ClinicalTrials GitHub repository and place it in the installed `trialbench/data/mesh-embeddings/` directory.

## Train

```bash
PYTHONPATH=src python scripts/train_trialbench_model.py --task outcome --phase All --download
```

Other useful TrialBench tasks include `mortality_rate_yn`, `serious_adverse_rate_yn`, `patient_dropout_rate_yn`, `failure_reason`, `duration`, `dose`, and `dose_cls`.

The trained artifact is saved to `models/trialfusion_light.joblib`.

## Run The Front Screen

```bash
PYTHONPATH=src flask --app app run --host 127.0.0.1 --port 5050
```

Open `http://127.0.0.1:5050`.
