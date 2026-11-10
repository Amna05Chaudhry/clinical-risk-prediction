"""SHAP explanations mapped back from one-hot columns to the original features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from .features import feature_map


class Explainer:
    """Wraps a fitted pipeline and returns SHAP values per *original* feature.

    One-hot columns produced from a categorical feature are summed back into
    that feature, so 'Chest pain type' gets one contribution, not four.
    Values are in the model's margin space (log-odds for LR / XGBoost,
    probability for the sklearn trees); only sign and size matter for display.
    """

    def __init__(self, pipe: Pipeline, background: pd.DataFrame, columns: list[str]):
        self.pipe = pipe
        self.columns = list(columns)
        self.pre = pipe.named_steps["pre"]
        self.model = pipe.named_steps["model"]
        out_names = self.pre.get_feature_names_out()
        # "num__age" -> age ; "cat__cp_4.0" -> cp
        self.owner = []
        for n in out_names:
            body = n.split("__", 1)[1]
            owner = next(c for c in sorted(self.columns, key=len, reverse=True)
                         if body == c or body.startswith(c + "_"))
            self.owner.append(owner)
        bg = self.pre.transform(background[self.columns])
        if hasattr(self.model, "coef_"):
            self.kind = "linear"
            self._exp = shap.LinearExplainer(self.model, bg)
        else:
            self.kind = "tree"
            self._exp = shap.TreeExplainer(self.model)

    def _raw_shap(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.pre.transform(X[self.columns])
        sv = self._exp.shap_values(Xt)
        if isinstance(sv, list):  # old shap API: one array per class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, features, classes)
            sv = sv[:, :, 1]
        return sv

    def shap_by_feature(self, X: pd.DataFrame) -> pd.DataFrame:
        sv = self._raw_shap(X)
        df = pd.DataFrame(sv, columns=self.owner)
        return df.T.groupby(level=0).sum().T[self.columns]

    def explain_one(self, row: pd.DataFrame, task: str, top: int = 5) -> list[dict]:
        """Top contributing features for one patient, with a plain-language sentence."""
        contrib = self.shap_by_feature(row).iloc[0]
        fm = feature_map(task)
        order = contrib.abs().sort_values(ascending=False).index[:top]
        out = []
        for name in order:
            f = fm[name]
            val = row.iloc[0][name]
            out.append({
                "feature": name,
                "label": f.label,
                "value": val,
                "value_text": describe_value(f, val),
                "shap": float(contrib[name]),
                "direction": "raises" if contrib[name] > 0 else "lowers",
                "sentence": sentence(f, val, float(contrib[name])),
            })
        return out


def describe_value(f, val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "not recorded"
    if f.kind == "cat":
        key = val
        if key not in f.options and isinstance(val, float) and float(val).is_integer():
            key = int(val)
        return str(f.options.get(key, val))
    v = float(val)
    txt = f"{v:g}" if f.step < 1 else f"{v:.0f}"
    if f.ordinal:  # grades / counts: "2 of 5" reads better than "2 grade 0-5"
        top = f.unit.split("-")[-1] if "-" in f.unit else ""
        return f"{txt} of {top}" if top else txt
    return f"{txt} {f.unit}".strip()


def range_note(f, val) -> str:
    if f.kind != "num" or f.normal is None or val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    lo, hi = f.normal
    v = float(val)
    if v < lo:
        return "below the typical range"
    if v > hi:
        return "above the typical range"
    return "within the typical range"


def sentence(f, val, s: float) -> str:
    verb = "pushes the risk up" if s > 0 else "pulls the risk down"
    note = range_note(f, val)
    note = f" ({note})" if note else ""
    return f"{f.label} = {describe_value(f, val)}{note} {verb}."
