"""Load and clean the raw UCI files into (X, y) with consistent dtypes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _missing(task: str) -> str:
    return f"{RAW / (task + '.csv')} not found - run `python scripts/download_data.py` first."


def load_ckd() -> tuple[pd.DataFrame, pd.Series]:
    path = RAW / "ckd.csv"
    if not path.exists():
        raise FileNotFoundError(_missing("ckd"))
    df = pd.read_csv(path, na_values=["?", "\t?"])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "?": np.nan})
    # target: 'ckd' / 'notckd' (a few raw rows have stray tabs or periods)
    y = df["class"].str.replace(r"[^a-z]", "", regex=True).map({"ckd": 1, "notckd": 0})
    X = df.drop(columns=["class"])
    for f in FEATURES["ckd"]:
        if f.kind == "num":
            X[f.name] = pd.to_numeric(X[f.name], errors="coerce")
        else:
            col = X[f.name].astype(object)
            X[f.name] = col.where(col.isin(list(f.options)), np.nan).astype(object)
    keep = y.notna()
    return X.loc[keep].reset_index(drop=True), y.loc[keep].astype(int).reset_index(drop=True)


def load_heart() -> tuple[pd.DataFrame, pd.Series]:
    path = RAW / "heart.csv"
    if not path.exists():
        raise FileNotFoundError(_missing("heart"))
    df = pd.read_csv(path, na_values=["?"])
    y = (df["num"] > 0).astype(int)  # 0 = no disease, 1-4 = disease present
    X = df.drop(columns=["num"])
    for f in FEATURES["heart"]:
        X[f.name] = pd.to_numeric(X[f.name], errors="coerce").astype(float)
        if f.kind == "cat":
            X[f.name] = X[f.name].where(X[f.name].isin(list(f.options)), np.nan)
    return X, y


def load(task: str) -> tuple[pd.DataFrame, pd.Series]:
    return {"ckd": load_ckd, "heart": load_heart}[task]()
