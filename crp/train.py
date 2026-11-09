"""Train, compare and explain both models.

    python -m crp.train            # both tasks
    python -m crp.train --task ckd

For each task:
  1. clean + de-duplicate, then an 80/20 stratified split (test set is touched once, at the end)
  2. (CKD only) feature selection 24 -> 12 on the training split only
  3. stratified 5-fold CV of LR / Decision Tree / Random Forest / XGBoost on the training split
  4. pick the model with the best mean CV ROC-AUC, refit on the full training split
  5. evaluate every model once on the held-out test set
  6. leakage / separability checks, SHAP global importance, figures, sample patients
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold, cross_validate, permutation_test_score, train_test_split,
)
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from .data import load  # noqa: E402
from .explain import Explainer  # noqa: E402
from .features import FEATURES, feature_map  # noqa: E402
from .pipeline import MODEL_NAMES, SEED, make_pipeline  # noqa: E402

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # also silences joblib worker processes
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGS = ROOT / "docs" / "figures"
EXAMPLES = ROOT / "examples"

TEAL = "#0F766E"
PALETTE = {"Logistic Regression": "#0F766E", "Decision Tree": "#B45309",
           "Random Forest": "#4F46E5", "XGBoost": "#BE185D"}
N_SELECT_CKD = 12
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return str(o)
    raise TypeError(type(o))


def cv_compare(task, X, y, columns) -> dict:
    res = {}
    for name in MODEL_NAMES:
        cvr = cross_validate(make_pipeline(task, columns, name), X[columns], y, cv=CV, scoring=SCORING, n_jobs=1)
        res[name] = {m: {"mean": float(cvr[f"test_{m}"].mean()), "std": float(cvr[f"test_{m}"].std())}
                     for m in SCORING}
    return res


def select_features_ckd(X: pd.DataFrame, y: pd.Series, k: int) -> tuple[list[str], dict]:
    """Permutation importance (log-loss) of a Random Forest, averaged over the 5 CV folds.

    Fit on each training fold, permute one *raw* column at a time on the matching
    validation fold, and measure how much the log-loss gets worse. Log-loss is
    used instead of accuracy because accuracy saturates near 100% on this data,
    which would make almost every feature look useless. Only the training split
    is used, so the held-out test set plays no part in choosing features.
    """
    cols = list(X.columns)
    scores = []
    for tr, va in CV.split(X, y):
        pipe = make_pipeline("ckd", cols, "Random Forest").fit(X.iloc[tr], y.iloc[tr])
        pi = permutation_importance(pipe, X.iloc[va], y.iloc[va], scoring="neg_log_loss",
                                    n_repeats=20, random_state=SEED, n_jobs=1)
        scores.append(pi.importances_mean)
    imp = pd.Series(np.mean(scores, axis=0), index=cols).sort_values(ascending=False)
    return list(imp.index[:k]), {c: float(v) for c, v in imp.items()}


def pick_best(cv: dict) -> str:
    # best mean ROC-AUC; ties (within 0.002) broken by mean F1, then by lower AUC std
    best_auc = max(v["roc_auc"]["mean"] for v in cv.values())
    tied = [n for n, v in cv.items() if v["roc_auc"]["mean"] >= best_auc - 0.002]
    return max(tied, key=lambda n: (round(cv[n]["f1"]["mean"], 4), -cv[n]["roc_auc"]["std"]))


def test_metrics(y_true, proba, thr=0.5) -> dict:
    pred = (proba >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, pred)),
        "specificity": float(recall_score(1 - np.asarray(y_true), 1 - pred)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "roc": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
    }


def separability_checks(task, X_tr, y_tr, X_te, y_te, columns, best) -> dict:
    """Evidence that a near-perfect score is the data, not a leak."""
    out = {}
    # 1. exact duplicate patients shared by train and test
    tr_keys = set(map(tuple, X_tr[columns].astype(str).values))
    out["train_test_duplicate_rows"] = int(sum(tuple(r) in tr_keys for r in X_te[columns].astype(str).values))
    # 2. label-permutation test: shuffled labels should fall to chance
    score, perm_scores, p = permutation_test_score(
        make_pipeline(task, columns, best), X_tr[columns], y_tr, cv=CV, scoring="roc_auc",
        n_permutations=100, random_state=SEED, n_jobs=1)
    out["permutation_test"] = {"true_auc": float(score), "shuffled_auc_mean": float(perm_scores.mean()),
                               "shuffled_auc_std": float(perm_scores.std()), "p_value": float(p),
                               "n_permutations": 100}
    # 3. one-feature decision stumps: how far does a single threshold get you?
    stumps = {}
    for c in columns:
        if feature_map(task)[c].kind != "num":
            continue
        pipe = make_pipeline(task, [c], "Logistic Regression")
        pipe.steps[-1] = ("model", DecisionTreeClassifier(max_depth=1, random_state=SEED))
        acc = cross_validate(pipe, X_tr[[c]], y_tr, cv=CV, scoring="accuracy")["test_score"].mean()
        stumps[c] = float(acc)
    out["single_feature_stump_cv_accuracy"] = dict(sorted(stumps.items(), key=lambda kv: -kv[1]))
    # 4. missingness differs by class (a dataset artefact worth disclosing)
    miss = X_tr[columns].isna().any(axis=1)
    out["rows_with_missing_by_class"] = {"positive": float(miss[y_tr == 1].mean()),
                                         "negative": float(miss[y_tr == 0].mean())}
    return out


# ------------------------------------------------------------------ figures
def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "figure.dpi": 110})


def fig_roc(task, title, rocs: dict):
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, m in rocs.items():
        ax.plot(m["roc"]["fpr"], m["roc"]["tpr"], lw=2, color=PALETTE[name],
                label=f"{name} (AUC {m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#94A3B8", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{title}: ROC on held-out test set")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / f"{task}_roc.png")
    plt.close(fig)


def fig_cm(task, title, cm, labels):
    _style()
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0F172A")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"{title}: confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGS / f"{task}_confusion.png")
    plt.close(fig)


def fig_shap(task, title, imp: pd.Series, shap_df: pd.DataFrame, X: pd.DataFrame):
    _style()
    fm = feature_map(task)
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.38 * len(imp) + 1))
    ax.barh([fm[c].label for c in imp.index], imp.values, color=TEAL)
    ax.set(xlabel="mean |SHAP value|  (average impact on model output)", title=f"{title}: global feature importance")
    fig.tight_layout()
    fig.savefig(FIGS / f"{task}_shap_global.png")
    plt.close(fig)

    # beeswarm-style plot: one dot per test patient, coloured by feature value
    import shap
    Xn = X.copy()
    for c in Xn.columns:
        if fm[c].kind == "cat":
            Xn[c] = pd.Series(pd.Categorical(Xn[c], categories=list(fm[c].options)).codes, index=Xn.index)
            Xn[c] = Xn[c].replace(-1, np.nan)
    order = list(imp.index[::-1])
    plt.figure()
    shap.summary_plot(shap_df[order].values, Xn[order].astype(float).values,
                      feature_names=[fm[c].label for c in order], show=False, plot_size=(7, 0.4 * len(order) + 1.2))
    plt.title(f"{title}: SHAP values per test patient", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS / f"{task}_shap_beeswarm.png", dpi=110)
    plt.close("all")


# ------------------------------------------------------------------ main
def run(task: str) -> dict:
    title = {"ckd": "Chronic Kidney Disease", "heart": "Heart Disease"}[task]
    print(f"\n=== {title} ===")
    X, y = load(task)
    n_raw = len(X)
    full = pd.concat([X, y.rename("target")], axis=1)
    dup_mask = full.astype(str).duplicated()
    n_dup = int(dup_mask.sum())
    X, y = X.loc[~dup_mask].reset_index(drop=True), y.loc[~dup_mask].reset_index(drop=True)
    print(f"rows: {n_raw} raw, {n_dup} exact duplicates removed -> {len(X)}; positives {y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    all_cols = [f.name for f in FEATURES[task]]

    report = {
        "task": task, "title": title,
        "data": {"rows_raw": n_raw, "duplicates_removed": n_dup, "rows": len(X),
                 "n_features_raw": len(all_cols), "positive_rate": float(y.mean()),
                 "train_rows": len(X_tr), "test_rows": len(X_te),
                 "missing_cells_pct": float(X.isna().mean().mean() * 100)},
    }

    if task == "ckd":
        cols, importances = select_features_ckd(X_tr, y_tr, N_SELECT_CKD)
        print("selected:", cols)
        report["feature_selection"] = {
            "method": "Random-Forest permutation importance (log-loss), averaged over 5 CV folds of the training split",
            "k": N_SELECT_CKD, "selected": cols, "importance": importances,
            "cv_all_features": cv_compare(task, X_tr, y_tr, all_cols),
        }
    else:
        cols = all_cols
    report["features"] = cols

    cv = cv_compare(task, X_tr, y_tr, cols)
    report["cv"] = cv
    for n, v in cv.items():
        print(f"  CV {n:20s} acc {v['accuracy']['mean']:.3f}±{v['accuracy']['std']:.3f}  "
              f"f1 {v['f1']['mean']:.3f}±{v['f1']['std']:.3f}  auc {v['roc_auc']['mean']:.3f}±{v['roc_auc']['std']:.3f}")
    best = pick_best(cv)
    report["best_model"] = best
    print("best by CV ROC-AUC:", best)

    test = {}
    fitted = {}
    for name in MODEL_NAMES:
        pipe = make_pipeline(task, cols, name).fit(X_tr[cols], y_tr)
        fitted[name] = pipe
        test[name] = test_metrics(y_te, pipe.predict_proba(X_te[cols])[:, 1])
        print(f"  TEST {name:20s} acc {test[name]['accuracy']:.3f}  f1 {test[name]['f1']:.3f}  auc {test[name]['roc_auc']:.3f}")
    report["test"] = test

    report["checks"] = separability_checks(task, X_tr, y_tr, X_te, y_te, cols, best)
    pt = report["checks"]["permutation_test"]
    print(f"  permutation test: true AUC {pt['true_auc']:.3f} vs shuffled {pt['shuffled_auc_mean']:.3f} (p={pt['p_value']:.3f})")

    # SHAP on the held-out test set
    pipe = fitted[best]
    bg = X_tr[cols].sample(min(100, len(X_tr)), random_state=SEED)
    expl = Explainer(pipe, bg, cols)
    shap_te = expl.shap_by_feature(X_te[cols])
    imp = shap_te.abs().mean().sort_values(ascending=False)
    report["shap_global"] = {c: float(v) for c, v in imp.items()}
    report["shap_space"] = "log-odds" if best in ("Logistic Regression", "XGBoost") else "probability"

    # figures
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_roc(task, title, test)
    labels = ["No CKD", "CKD"] if task == "ckd" else ["No disease", "Disease"]
    fig_cm(task, title, test[best]["confusion_matrix"], labels)
    fig_shap(task, title, imp, shap_te, X_te[cols])

    # data ranges for the UI sliders (training split only)
    report["ranges"] = {c: {"min": float(X_tr[c].min()), "max": float(X_tr[c].max()),
                            "median": float(X_tr[c].median())}
                        for c in cols if feature_map(task)[c].kind == "num"}
    report["modes"] = {c: X_tr[c].mode().iloc[0] for c in cols if feature_map(task)[c].kind == "cat"}

    # sample patients from the held-out test set
    proba = pipe.predict_proba(X_te[cols])[:, 1]
    te = X_te[cols].copy()
    te["_p"], te["_y"] = proba, y_te.values
    complete = te.dropna()
    pool = complete if len(complete) >= 10 else te
    picks = [
        ("High-risk patient", pool[pool._y == 1].sort_values("_p").iloc[-1]),
        ("Low-risk patient", pool[pool._y == 0].sort_values("_p").iloc[0]),
        ("Borderline patient", pool.iloc[(pool._p - 0.5).abs().argsort().iloc[0]]),
    ]
    samples = []
    for name, r in picks:
        vals = {c: (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c], "item") else r[c])) for c in cols}
        samples.append({"name": name, "true_label": int(r["_y"]), "values": vals})
    EXAMPLES.mkdir(exist_ok=True)
    (EXAMPLES / f"{task}_samples.json").write_text(json.dumps(samples, indent=2, default=_jsonable))

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "columns": cols, "model_name": best, "background": bg, "task": task},
                MODELS / f"{task}.joblib", compress=3)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ckd", "heart", "all"], default="all")
    a = ap.parse_args()
    tasks = ["ckd", "heart"] if a.task == "all" else [a.task]
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "metrics.json"
    allrep = json.loads(path.read_text()) if path.exists() else {}
    for t in tasks:
        allrep[t] = run(t)
    path.write_text(json.dumps(allrep, indent=2, default=_jsonable))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
