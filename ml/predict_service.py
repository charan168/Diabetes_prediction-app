"""
Long-running prediction worker.

The Node backend spawns this process ONCE at startup. The model is loaded once, then the
worker reads one JSON object per line on stdin and answers with one JSON line on stdout:

    request : {"id": 1, "features": {"Glucose": 120, ...}}
    response: {"id": 1, "ok": true, "prediction": 0, "probability": 0.31}
              {"id": 1, "ok": false, "error": "message"}

Only the protocol goes to stdout; logs go to stderr.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH", Path(__file__).resolve().parent / "artifacts" / "diabetes_rf.joblib"
    )
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"], set(bundle["zero_means_missing"])


def to_row(features: dict, order: list[str], zero_missing: set[str]) -> pd.DataFrame:
    row = {}
    for name in order:
        value = features.get(name)
        if value is None:
            row[name] = np.nan
            continue
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name} must be a finite number")
        # Same convention as training: 0 means "not measured" for these columns.
        row[name] = np.nan if (name in zero_missing and value == 0) else value
    return pd.DataFrame([row], columns=order)


def main() -> None:
    model, order, zero_missing = load_bundle()
    log(f"model loaded from {MODEL_PATH}")
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            frame = to_row(req["features"], order, zero_missing)
            proba = float(model.predict_proba(frame)[0, 1])
            out = {
                "id": req_id,
                "ok": True,
                "prediction": int(proba >= 0.5),
                "probability": round(proba, 4),
            }
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            out = {"id": req_id, "ok": False, "error": str(exc)}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
