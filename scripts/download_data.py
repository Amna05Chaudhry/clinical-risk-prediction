"""Download the two public UCI datasets used by this project.

    python scripts/download_data.py

Writes:
    data/raw/ckd.csv    UCI Chronic Kidney Disease (id 336) - 400 rows, 24 features + class
    data/raw/heart.csv  UCI Heart Disease, Cleveland subset (id 45) - 303 rows, 13 features + num

Both come straight from the UCI Machine Learning Repository. The script first
tries the official `ucimlrepo` API and falls back to the direct file links.
"""
from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

CKD_ZIP = "https://archive.ics.uci.edu/static/public/336/chronic+kidney+disease.zip"
HEART_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)

CKD_COLS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod",
    "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet", "pe", "ane", "class",
]
HEART_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num",
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "clinical-risk-prediction/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


# ---------------------------------------------------------------- CKD
def _ckd_from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=336)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = [c.strip() for c in df.columns]
    return df


def _ckd_from_zip() -> pd.DataFrame:
    """Parse the original ARFF by hand (it has a few malformed rows: extra commas, tabs)."""
    zf = zipfile.ZipFile(io.BytesIO(_get(CKD_ZIP)))
    name = next(n for n in zf.namelist() if n.endswith("chronic_kidney_disease_full.arff"))
    text = zf.read(name).decode("utf-8", errors="ignore")
    body = text.split("@data", 1)[1]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        vals = [v.strip() for v in re.split(r",", line)]
        vals = [v for v in vals if v != ""]  # drop empty fields from double commas
        if len(vals) != len(CKD_COLS):
            continue
        rows.append(vals)
    return pd.DataFrame(rows, columns=CKD_COLS)


def download_ckd() -> pd.DataFrame:
    try:
        df = _ckd_from_ucimlrepo()
        src = "ucimlrepo"
    except Exception as e:  # noqa: BLE001
        print(f"  ucimlrepo failed ({e!r}); using direct ARFF link")
        df = _ckd_from_zip()
        src = "direct link"
    df = df.replace({"?": np.nan, "": np.nan})
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string").str.strip().str.replace("\t", "", regex=False)
    print(f"  CKD: {df.shape[0]} rows x {df.shape[1]} cols (via {src})")
    return df


# ---------------------------------------------------------------- Heart
def _heart_from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=45)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)


def _heart_from_url() -> pd.DataFrame:
    raw = _get(HEART_URL).decode()
    return pd.read_csv(io.StringIO(raw), header=None, names=HEART_COLS, na_values="?")


def download_heart() -> pd.DataFrame:
    try:
        df = _heart_from_ucimlrepo()
        src = "ucimlrepo"
    except Exception as e:  # noqa: BLE001
        print(f"  ucimlrepo failed ({e!r}); using direct link")
        df = _heart_from_url()
        src = "direct link"
    df = df[HEART_COLS]
    print(f"  Heart (Cleveland): {df.shape[0]} rows x {df.shape[1]} cols (via {src})")
    return df


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    print("Downloading UCI datasets ...")
    download_ckd().to_csv(RAW / "ckd.csv", index=False)
    download_heart().to_csv(RAW / "heart.csv", index=False)
    print(f"Saved to {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
