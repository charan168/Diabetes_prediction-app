# Diabetes Risk Prediction — Demo App with Model Evaluation Dashboard

React · Node/Express · Python/scikit-learn · MongoDB

> **For demonstration purposes only. Predictions are not medical advice** and must not be used for
> diagnosis or treatment. The model is small, trained on a public research dataset, and has not been
> clinically validated.

A patient-attribute form returns a diabetes-risk prediction and probability from a RandomForest model.
Every prediction is logged to MongoDB. A password-protected dashboard shows the latest 10 predictions,
aggregate counts, and the model's **measured held-out test metrics**.

```
React (Vite, :5173) ──/api──▶ Express (:5000) ──stdin/stdout JSON──▶ Python worker (model loaded once)
                                   │
                                   └──────────────▶ MongoDB
```

## Project structure

```
disease-prediction-app/
├── ml/
│   ├── data/diabetes.csv          # Pima Indians Diabetes dataset (768 rows, public)
│   ├── train_model.py             # training + evaluation script
│   ├── predict_service.py         # long-running worker; loads the model once
│   ├── artifacts/
│   │   ├── diabetes_rf.joblib     # trained model (sklearn Pipeline)
│   │   └── metrics.json           # held-out test metrics written by train_model.py
│   └── requirements.txt
├── backend/
│   ├── src/ (server, app, config, routes/, middleware/, models/, services/, utils/)
│   ├── tests/api.test.js          # integration tests (real model + real MongoDB)
│   └── .env.example
└── frontend/
    └── src/ (pages/, components/, api.js, fields.js, App.test.jsx)
```

## Prerequisites

- Node.js **20.19+** (or 22.12+) — required by Vite 8
- Python **3.10+**
- MongoDB running locally (`mongodb://127.0.0.1:27017`) or a MongoDB Atlas URI

## Setup and run

```bash
# 1. Python dependencies (use a venv if you like)
pip install -r ml/requirements.txt

# 2. (Optional) retrain — a trained model is already included in ml/artifacts/
python ml/train_model.py          # ~1 min; rewrites diabetes_rf.joblib and metrics.json

# 3. Backend
cd backend
cp .env.example .env              # then edit DASHBOARD_PASSWORD and MONGODB_URI
npm install
npm start                         # http://localhost:5000

# 4. Frontend (new terminal)
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

Open <http://localhost:5173>, use **Quick fill → Sample A** and press **Predict**, then open
**Dashboard** and sign in with the `DASHBOARD_USER` / `DASHBOARD_PASSWORD` from `backend/.env`.

> The saved model was trained with scikit-learn 1.9.1. Install the pinned versions in
> `ml/requirements.txt`, or re-run `python ml/train_model.py` if you use a different version.

## Environment variables

**Backend (`backend/.env`)**

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `MONGODB_URI` | yes | – | MongoDB connection string |
| `DASHBOARD_USER` | yes | – | Basic-auth username for dashboard endpoints |
| `DASHBOARD_PASSWORD` | yes | – | Basic-auth password (**change it**) |
| `PORT` | no | `5000` | API port |
| `CORS_ORIGIN` | no | `http://localhost:5173` | Allowed browser origin(s), comma-separated |
| `PYTHON_BIN` | no | `python3` | Python executable (use `python` on Windows, or a venv path) |
| `MODEL_PATH` | no | `ml/artifacts/diabetes_rf.joblib` | Override model file |
| `NODE_ENV` | no | `development` | |

**Frontend (`frontend/.env`)** — optional: `VITE_API_URL` (default `/api`, proxied to `:5000` by Vite;
set `VITE_PROXY_TARGET` if the backend runs elsewhere).

## API

All responses share one envelope: `{ "success": true, "data": … }` or
`{ "success": false, "error": { "code", "message", "details?" } }`.

| Method & path | Auth | Description |
|---|---|---|
| `GET /api/health` | none | Database + model readiness |
| `POST /api/predict` | none | Validate input, predict, save to MongoDB, return result (201) |
| `GET /api/predictions` | Basic | Latest 10 predictions, newest first |
| `GET /api/metrics` | Basic | Totals, positive/negative counts and percentages |
| `GET /api/model-evaluation` | Basic | Held-out test metrics from `ml/artifacts/metrics.json` |

```bash
curl -X POST http://localhost:5000/api/predict -H 'Content-Type: application/json' -d '{
  "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
  "Insulin": null, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50 }'

curl -u admin:your-password http://localhost:5000/api/metrics
```

**Input fields and validation** (server-enforced; the form mirrors the same rules):
`Pregnancies` 0–20 (integer), `Glucose` 1–500, `BloodPressure` 1–250, `BMI` 10–80,
`DiabetesPedigreeFunction` 0.01–3, `Age` 1–120 (integer) are required.
`SkinThickness` 0–100 and `Insulin` 0–900 are optional (`null`/omitted = not measured; imputed by the
model pipeline). Values must be JSON numbers; unknown fields are ignored; bodies over 10 KB are rejected.

Error codes: `VALIDATION_ERROR` (400, with per-field `details`), `INVALID_JSON` (400),
`UNAUTHORIZED` (401), `NOT_FOUND` (404), `PAYLOAD_TOO_LARGE` (413), `MODEL_UNAVAILABLE` /
`DB_UNAVAILABLE` (503), `INTERNAL_ERROR` (500). If MongoDB is unreachable, `/api/predict` returns 503
rather than returning an un-logged prediction.

## Machine learning

**Dataset:** Pima Indians Diabetes (768 rows; 8 numeric features; `Outcome` = 1 for diabetes, 34.9% positive).

**Preprocessing and training** (`ml/train_model.py`):
1. Drop duplicate rows (none found).
2. In `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, a value of `0` is physiologically
   impossible and encodes “missing”; it is converted to `NaN`. The prediction worker applies the same
   rule, so a `0` or blank for these fields is handled consistently.
3. Stratified 80/20 split (`random_state=42`): 614 train / 154 test.
4. `Pipeline`: `SimpleImputer(median)` → `RandomForestClassifier(class_weight="balanced")`. The imputer
   is inside the pipeline, so medians come from training data only.
5. Hyperparameters chosen by 5-fold stratified CV on the **training set only** (ROC-AUC). Selected:
   200 trees, `max_depth=4`, `min_samples_leaf=5`.
6. The final model is scored once on the held-out test set; those numbers are saved to `metrics.json`.

**Measured results on the 154-row held-out test set** (decision threshold 0.5):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.734 | 0.592 | 0.778 | 0.672 | 0.816 |

Confusion matrix — TN 71, FP 29, FN 12, TP 42. (5-fold CV ROC-AUC on the training set: 0.840.)

These come from a small dataset and a single split, so expect noticeable variance; they are not claims
about real-world clinical performance. The dashboard reads the same `metrics.json`, so retraining
updates it automatically.

**How the model is served:** the backend spawns `ml/predict_service.py` once at startup; the model is
loaded once and kept in memory, and requests are exchanged as JSON lines. If the worker crashes it is
restarted automatically.

## Tests

```bash
cd backend  && npm test     # 16 integration tests: predict, validation, auth, DB logging, metrics, edge cases
cd frontend && npm test     # 10 tests: form validation, result display, dashboard login/data, navigation
```

Backend tests use `TEST_MONGODB_URI` (default `mongodb://127.0.0.1:27017/disease_prediction_test`) and
clear that test database. They never touch your main database.

## Notes and limitations

- Dashboard auth is HTTP Basic with credentials from the environment, stored in `sessionStorage` in the
  browser. Fine for a demo; **use HTTPS** if deployed and add rate limiting / a real auth system for
  anything beyond a demo. `POST /api/predict` is intentionally public.
- Dataset: Pima Indians Diabetes data was collected from adult women of Pima heritage (age ≥ 21), so the
  model should not be assumed to generalise to other populations.
- Patient inputs are stored in MongoDB without identifiers. Do not enter real patient data in a demo.
