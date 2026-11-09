"""Smoke tests: the saved models load, predict and explain the bundled sample patients.

    python -m pytest -q      (or: python tests/test_smoke.py)
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crp.explain import Explainer  # noqa: E402
from crp.features import feature_map  # noqa: E402


def _row(task, cols, values):
    fm = feature_map(task)
    row = pd.DataFrame([values])[cols]
    for c in cols:
        row[c] = row[c].astype(float) if (fm[c].kind == "num" or task == "heart") else row[c].astype(object)
    return row


def test_models_predict_and_explain():
    for task in ("ckd", "heart"):
        b = joblib.load(ROOT / "models" / f"{task}.joblib")
        expl = Explainer(b["pipeline"], b["background"], b["columns"])
        samples = json.loads((ROOT / "examples" / f"{task}_samples.json").read_text())
        probs = {}
        for s in samples:
            row = _row(task, b["columns"], s["values"])
            p = float(b["pipeline"].predict_proba(row)[0, 1])
            assert 0.0 <= p <= 1.0
            factors = expl.explain_one(row, task, top=5)
            assert len(factors) == 5 and all(f["sentence"] for f in factors)
            probs[s["name"]] = p
        assert probs["High-risk patient"] > probs["Low-risk patient"]
        print(task, {k: round(v, 3) for k, v in probs.items()})


if __name__ == "__main__":
    test_models_predict_and_explain()
    print("ok")
