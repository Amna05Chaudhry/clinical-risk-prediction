"""The shared sklearn pipeline: impute -> encode -> scale -> model."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .features import feature_map

SEED = 42

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]


def make_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, C=1.0)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=4, random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=4, random_state=SEED,
        )
    raise ValueError(name)


def make_preprocessor(task: str, columns: list[str]) -> ColumnTransformer:
    """Numeric: median impute + standardise. Categorical: mode impute + one-hot.

    Everything is fitted inside the Pipeline, so in cross-validation the imputer
    and scaler only ever see the training fold (no information from the
    validation fold or the test set leaks in).
    """
    fm = feature_map(task)
    num = [c for c in columns if fm[c].kind == "num"]
    cat = [c for c in columns if fm[c].kind == "cat"]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
        ]), cat))
    return ColumnTransformer(transformers, verbose_feature_names_out=True)


def make_pipeline(task: str, columns: list[str], model_name: str) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(task, columns)), ("model", make_model(model_name))])
