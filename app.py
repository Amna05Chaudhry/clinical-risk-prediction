"""Clinical Risk Prediction - Streamlit app.

    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crp.explain import Explainer, describe_value
from crp.features import FEATURES, feature_map
from crp.pipeline import MODEL_NAMES

ROOT = Path(__file__).resolve().parent
TEAL, RED, AMBER, GREEN = "#0F766E", "#DC2626", "#D97706", "#16A34A"
MODEL_COLORS = {"Logistic Regression": "#0F766E", "Decision Tree": "#B45309",
                "Random Forest": "#4F46E5", "XGBoost": "#BE185D"}
TITLES = {"ckd": "Chronic Kidney Disease", "heart": "Heart Disease"}

st.set_page_config(page_title="Clinical Risk Prediction", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea, label {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.block-container { padding-top: 1.6rem; max-width: 1250px; }
.hero {
  background: linear-gradient(120deg, #0F766E 0%, #115E59 55%, #134E4A 100%);
  border-radius: 18px; padding: 1.5rem 1.8rem; color: #F0FDFA; margin-bottom: 1rem;
  box-shadow: 0 8px 24px rgba(15,118,110,.18);
}
.hero h1 { color: #FFFFFF; font-size: 2rem; font-weight: 800; margin: 0 0 .25rem 0; padding: 0; letter-spacing: -.02em; }
.hero p { margin: 0; font-size: 1.02rem; opacity: .92; }
.pill { display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .78rem;
  font-weight: 600; margin: .7rem .35rem 0 0; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); }
.pill.warn { background: #FEF3C7; color: #92400E; border-color: #FCD34D; }
.card { border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 1rem 1.15rem;
  background: rgba(128,128,128,.04); margin-bottom: .8rem; }
.factor { display: flex; gap: .65rem; align-items: flex-start; padding: .45rem 0;
  border-bottom: 1px dashed rgba(128,128,128,.25); font-size: .95rem; }
.factor:last-child { border-bottom: none; }
.arrow { font-weight: 800; min-width: 1.2rem; text-align: center; }
.up { color: #DC2626; } .down { color: #0F766E; }
.risk-band { font-size: 1.15rem; font-weight: 700; margin: -.6rem 0 .2rem 0; text-align: center; }
.muted { opacity: .72; font-size: .86rem; }
.disclaimer { border-left: 4px solid #D97706; background: rgba(217,119,6,.08); padding: .6rem .9rem;
  border-radius: 8px; font-size: .88rem; margin: .4rem 0 1rem 0; }
div[data-testid="stForm"] { border-radius: 14px; border: 1px solid rgba(128,128,128,.22); }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: .45rem .9rem; border-radius: 10px 10px 0 0; }
.stButton > button { border-radius: 10px; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ resources
@st.cache_data
def load_report() -> dict:
    p = ROOT / "reports" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource
def load_model(task: str):
    bundle = joblib.load(ROOT / "models" / f"{task}.joblib")
    expl = Explainer(bundle["pipeline"], bundle["background"], bundle["columns"])
    return bundle, expl


@st.cache_data
def load_samples(task: str) -> list[dict]:
    p = ROOT / "examples" / f"{task}_samples.json"
    return json.loads(p.read_text()) if p.exists() else []


REPORT = load_report()
if not REPORT or not all((ROOT / "models" / f"{t}.joblib").exists() for t in ("ckd", "heart")):
    st.error("Trained models not found. Run `python scripts/download_data.py` and then `python -m crp.train`.")
    st.stop()


# ------------------------------------------------------------------ header
st.markdown("""
<div class="hero">
  <h1>🩺 Clinical Risk Prediction</h1>
  <p>Explainable machine-learning risk scores for chronic kidney disease and heart disease, trained on public UCI data.</p>
  <span class="pill">scikit-learn pipeline</span><span class="pill">4 models · stratified 5-fold CV</span>
  <span class="pill">SHAP explanations</span><span class="pill warn">⚠ Research demo, not for clinical use</span>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def wkey(task, name):
    return f"{task}__{name}"


def to_widget_value(f, v):
    if f.kind == "cat":
        if v is None:
            return list(f.options)[0]
        if isinstance(next(iter(f.options)), int):
            return int(v)
        return v
    return float(v)


def apply_sample(task, values: dict):
    fm = feature_map(task)
    for name, v in values.items():
        if v is not None:
            st.session_state[wkey(task, name)] = to_widget_value(fm[name], v)


def reset_defaults(task):
    rep = REPORT[task]
    fm = feature_map(task)
    for c in rep["features"]:
        if fm[c].kind == "num":
            st.session_state[wkey(task, c)] = default_num(fm[c], rep["ranges"][c])
        else:
            st.session_state[wkey(task, c)] = to_widget_value(fm[c], rep["modes"][c])


def default_num(f, r):
    v = r["median"]
    if f.normal is not None and not f.ordinal:  # start inside the typical range where we have one
        lo, hi = f.normal
        v = min(max(v, lo), hi)
    return float(round(v / f.step) * f.step) if f.step >= 1 else float(round(v, 3))


def widget(task, f, rep):
    key = wkey(task, f.name)
    norm = ""
    if f.normal is not None:
        lo, hi = f.normal
        norm = f"Typical range: {lo:g}–{hi:g} {f.unit}".strip() if lo != hi else f"Typical: {lo:g}"
    help_txt = " ".join(x for x in [f.help, norm] if x) or None
    label = f"{f.label} ({f.unit})" if f.unit else f.label
    if f.kind == "cat":
        opts = list(f.options)
        if key not in st.session_state:
            st.session_state[key] = to_widget_value(f, rep["modes"][f.name])
        return st.selectbox(label, opts, format_func=lambda o: f.options[o], key=key, help=help_txt)
    r = rep["ranges"][f.name]
    lo, hi = float(np.floor(r["min"])), float(np.ceil(r["max"]))
    if f.step < 1:
        lo, hi = float(r["min"]), float(r["max"])
    if key not in st.session_state:
        st.session_state[key] = default_num(f, r)
    st.session_state[key] = float(min(max(st.session_state[key], lo), hi))
    if (hi - lo) / f.step <= 250:
        return st.slider(label, lo, hi, step=float(f.step), key=key, help=help_txt)
    return st.number_input(label, lo, hi, step=float(f.step), key=key, help=help_txt)


def gauge(p: float):
    color = GREEN if p < 0.3 else AMBER if p < 0.7 else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 46}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [{"range": [0, 30], "color": "rgba(22,163,74,.16)"},
                      {"range": [30, 70], "color": "rgba(217,119,6,.16)"},
                      {"range": [70, 100], "color": "rgba(220,38,38,.16)"}],
            "threshold": {"line": {"color": "#64748B", "width": 2}, "thickness": 0.8, "value": 50},
        }))
    fig.update_layout(height=250, margin=dict(l=25, r=25, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def factor_chart(factors):
    f = list(reversed(factors))
    fig = go.Figure(go.Bar(
        x=[x["shap"] for x in f], y=[x["label"] for x in f], orientation="h",
        marker_color=[RED if x["shap"] > 0 else TEAL for x in f],
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.update_layout(height=60 + 38 * len(f), margin=dict(l=0, r=10, t=10, b=30),
                      xaxis_title="← lowers risk     contribution     raises risk →",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.add_vline(x=0, line_color="#94A3B8", line_width=1)
    return fig


# ------------------------------------------------------------------ prediction tab
GROUPS = {
    "ckd": [("Urine tests", ["sg", "al", "su", "rbc", "pc", "pcc", "ba"]),
            ("Blood tests", ["bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]),
            ("Patient & history", ["age", "bp", "htn", "dm", "cad", "appet", "pe", "ane"])],
    "heart": [("Patient", ["age", "sex", "cp", "trestbps", "chol", "fbs"]),
              ("ECG & exercise test", ["restecg", "thalach", "exang", "oldpeak", "slope"]),
              ("Imaging", ["ca", "thal"])],
}


def predict_tab(task: str):
    rep = REPORT[task]
    bundle, expl = load_model(task)
    cols = bundle["columns"]
    fm = feature_map(task)

    t = rep["test"][rep["best_model"]]
    intro = {
        "ckd": f"Uses the **{len(cols)} most informative of 24** routine urine and blood measurements "
               f"(chosen automatically, see Model card).",
        "heart": "Uses the 13 standard Cleveland clinic variables: symptoms, vitals, ECG, exercise test and imaging.",
    }[task]
    st.markdown(f"{intro} Model: **{rep['best_model']}** · held-out test ROC-AUC **{t['roc_auc']:.3f}**, "
                f"accuracy **{t['accuracy']:.1%}** (n={rep['data']['test_rows']}).")

    samples = load_samples(task)
    healthy = next((s for s in samples if s["name"] == "Low-risk patient"), None)
    if healthy and f"{task}__init" not in st.session_state:  # open on a healthy patient, not a mix of medians
        apply_sample(task, healthy["values"])
        st.session_state[f"{task}__init"] = True
    bcols = st.columns(len(samples) + 2)
    bcols[0].markdown("<div class='muted' style='padding-top:.45rem'>Try a sample patient:</div>", unsafe_allow_html=True)
    for i, s in enumerate(samples):
        bcols[i + 1].button(s["name"], key=f"{task}_sample_{i}", on_click=apply_sample, args=(task, s["values"]),
                            use_container_width=True,
                            help=f"Real patient from the held-out test set (true label: "
                                 f"{'positive' if s['true_label'] else 'negative'})")
    bcols[-1].button("Dataset medians", key=f"{task}_reset", on_click=reset_defaults, args=(task,),
                     use_container_width=True, help="Set every input to the training-set median / most common value")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        for gname, gcols in GROUPS[task]:
            present = [c for c in gcols if c in cols]
            if not present:
                continue
            with st.container(border=True):
                st.markdown(f"**{gname}**")
                c1, c2 = st.columns(2)
                for i, c in enumerate(present):
                    with (c1 if i % 2 == 0 else c2):
                        widget(task, fm[c], rep)

    values = {c: st.session_state[wkey(task, c)] for c in cols}
    row = pd.DataFrame([values])[cols]
    for c in cols:
        if fm[c].kind == "num" or task == "heart":
            row[c] = row[c].astype(float)
        else:
            row[c] = row[c].astype(object)

    with right:
        with st.spinner("Scoring patient…"):
            p = float(bundle["pipeline"].predict_proba(row)[:, 1][0])
            factors = expl.explain_one(row, task, top=5)
        band = ("Low", GREEN) if p < 0.3 else ("Moderate", AMBER) if p < 0.7 else ("High", RED)
        with st.container(border=True):
            st.markdown(f"**Estimated probability of {TITLES[task].lower()}**")
            st.plotly_chart(gauge(p), use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"<div class='risk-band' style='color:{band[1]}'>{band[0]} risk</div>"
                        f"<div class='muted' style='text-align:center'>Bands: &lt;30% low · 30–70% moderate · "
                        f"&gt;70% high. Classified positive at ≥50%.</div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Why? Top factors for this patient**")
            html = "".join(
                f"<div class='factor'><span class='arrow {'up' if f['shap'] > 0 else 'down'}'>"
                f"{'▲' if f['shap'] > 0 else '▼'}</span><span>{f['sentence']}</span></div>" for f in factors)
            st.markdown(html, unsafe_allow_html=True)
            st.plotly_chart(factor_chart(factors), use_container_width=True, config={"displayModeBar": False})
            st.caption(f"SHAP values ({rep['shap_space']} scale) from the {rep['best_model']} model. "
                       "They show how each input moved this prediction away from the average patient; "
                       "they describe the model, not the patient's biology.")
    st.markdown("<div class='disclaimer'>⚠ <b>Research demo, not for clinical use.</b> Trained on a few hundred "
                "historical patients from public datasets; it has not been validated and must not guide diagnosis "
                "or treatment.</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ model card
def fmt(m):
    return f"{m['mean']:.3f} ± {m['std']:.3f}"


def model_card():
    task = st.radio("Dataset", ["ckd", "heart"], format_func=lambda t: TITLES[t], horizontal=True,
                    key="card_task", label_visibility="collapsed")
    rep = REPORT[task]
    d = rep["data"]
    fm = feature_map(task)
    best = rep["best_model"]
    t = rep["test"][best]

    m = st.columns(5)
    m[0].metric("Deployed model", best)
    m[1].metric("Test ROC-AUC", f"{t['roc_auc']:.3f}")
    m[2].metric("Test accuracy", f"{t['accuracy']:.1%}")
    m[3].metric("Sensitivity", f"{t['sensitivity']:.1%}")
    m[4].metric("Specificity", f"{t['specificity']:.1%}")

    st.markdown("#### Cross-validation (training split, stratified 5-fold, mean ± std)")
    cv = pd.DataFrame({n: {"Accuracy": fmt(v["accuracy"]), "F1": fmt(v["f1"]), "ROC-AUC": fmt(v["roc_auc"])}
                       for n, v in rep["cv"].items()}).T
    st.dataframe(cv, use_container_width=True)
    st.markdown("#### Held-out test set (each model evaluated once)")
    tt = pd.DataFrame({n: {"Accuracy": f"{v['accuracy']:.3f}", "F1": f"{v['f1']:.3f}",
                           "ROC-AUC": f"{v['roc_auc']:.3f}", "Sensitivity": f"{v['sensitivity']:.3f}",
                           "Specificity": f"{v['specificity']:.3f}"}
                       for n, v in rep["test"].items()}).T
    st.dataframe(tt, use_container_width=True)
    st.caption(f"The deployed model is the one with the best mean CV ROC-AUC ({best}); "
               "it is chosen before looking at the test set.")

    c1, c2, c3 = st.columns([1.2, 0.9, 1.2])
    with c1:
        fig = go.Figure()
        for n in MODEL_NAMES:
            r = rep["test"][n]
            fig.add_trace(go.Scatter(x=r["roc"]["fpr"], y=r["roc"]["tpr"], mode="lines",
                                     name=f"{n} ({r['roc_auc']:.3f})", line=dict(width=2.5, color=MODEL_COLORS[n])))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94A3B8"),
                                 showlegend=False))
        fig.update_layout(title="ROC curves (test set)", height=380, xaxis_title="False positive rate",
                          yaxis_title="True positive rate", legend=dict(x=0.35, y=0.05, font=dict(size=11)),
                          margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cm = np.array(t["confusion_matrix"])
        labels = ["No CKD", "CKD"] if task == "ckd" else ["No disease", "Disease"]
        fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=[[0, "#F0FDFA"], [1, TEAL]],
                                   showscale=False, text=cm, texttemplate="%{text}", textfont={"size": 22}))
        fig.update_layout(title=f"Confusion matrix ({best})", height=380, xaxis_title="Predicted",
                          yaxis_title="Actual", yaxis_autorange="reversed",
                          margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        imp = pd.Series(rep["shap_global"]).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[fm[c].label for c in imp.index], orientation="h",
                               marker_color=TEAL))
        fig.update_layout(title="Global importance (mean |SHAP|, test set)", height=380,
                          margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    bees = ROOT / "docs" / "figures" / f"{task}_shap_beeswarm.png"
    if bees.exists():
        with st.expander("SHAP beeswarm: every test patient, coloured by feature value"):
            st.image(str(bees))

    st.markdown("#### Dataset")
    if task == "ckd":
        st.markdown(
            f"""**UCI Chronic Kidney Disease** (Rubini, Soundarapandian & Eswaran, 2015), collected at Apollo Hospitals,
Tamil Nadu, India over ~2 months. {d['rows_raw']} patients, 24 clinical features (11 numeric, 13 categorical),
{d['positive_rate']:.0%} CKD. About {d['missing_cells_pct']:.0f}% of cells are missing; they are imputed inside the
pipeline (median / most frequent), fitted on training folds only. Split: {d['train_rows']} train / {d['test_rows']} test,
stratified. {d['duplicates_removed']} exact duplicate rows removed before splitting.""")
        fs = rep["feature_selection"]
        st.markdown("#### Feature selection: 24 → 12")
        st.markdown(
            "Method: Random-Forest **permutation importance** measured with **log-loss** on the validation part of each "
            "of the 5 CV folds of the **training split only**, averaged, top 12 kept. Log-loss is used because accuracy "
            "is already ~100% and would rate almost every feature as useless. The 12 kept features are the classic CKD markers "
            "(hemoglobin, urine specific gravity, creatinine, albumin, hypertension, diabetes…); the dropped ones (white cell "
            "count, potassium, bacteria, coronary disease…) add almost nothing, and CV performance is unchanged (table on the right).")
        sel = pd.DataFrame({"Feature": [fm[c].label for c in fs["importance"]],
                            "Importance": [round(v, 4) for v in fs["importance"].values()],
                            "Selected": ["✅" if c in fs["selected"] else "" for c in fs["importance"]]})
        cc1, cc2 = st.columns([1, 1.2])
        cc1.dataframe(sel, use_container_width=True, hide_index=True, height=300)
        all_cv = pd.DataFrame({n: {"ROC-AUC (24 feat.)": fmt(fs["cv_all_features"][n]["roc_auc"]),
                                   "ROC-AUC (12 feat.)": fmt(rep["cv"][n]["roc_auc"]),
                                   "Acc (24)": fmt(fs["cv_all_features"][n]["accuracy"]),
                                   "Acc (12)": fmt(rep["cv"][n]["accuracy"])} for n in MODEL_NAMES}).T
        cc2.markdown("CV with all 24 vs the selected 12 features:")
        cc2.dataframe(all_cv, use_container_width=True)
    else:
        st.markdown(
            f"""**UCI Heart Disease, Cleveland Clinic subset** (Detrano et al., 1989). {d['rows_raw']} patients referred
for coronary angiography, 13 features. Target: any angiographic narrowing > 50% (`num` > 0), {d['positive_rate']:.0%}
positive. {d['duplicates_removed']} duplicate rows found. We deliberately did **not** use the popular 1,025-row
Kaggle "heart.csv": it is the same ~300 patients copied about three times, so a random split puts copies of test
patients in the training set and inflates accuracy to ~98–100%. Split: {d['train_rows']} train / {d['test_rows']} test, stratified.""")

    ck = rep["checks"]
    st.markdown("#### Is a high score real? Leakage & separability checks")
    pt = ck["permutation_test"]
    stumps = ck["single_feature_stump_cv_accuracy"]
    top_stumps = list(stumps.items())[:4]
    st.markdown(
        f"""- **Train/test overlap:** {ck['train_test_duplicate_rows']} test patients are exact copies of a training patient.
- **Label-shuffle test:** with the labels randomly permuted {pt['n_permutations']}×, CV ROC-AUC drops from
  **{pt['true_auc']:.3f}** to **{pt['shuffled_auc_mean']:.3f} ± {pt['shuffled_auc_std']:.3f}** (chance), p = {pt['p_value']:.3f}.
  A leaking pipeline would still score well on shuffled labels.
- **One-threshold rules (decision stumps, CV accuracy):** """ +
        ", ".join(f"{fm[c].label} **{a:.1%}**" for c, a in top_stumps) + ".")
    if task == "ckd":
        miss = ck["rows_with_missing_by_class"]
        st.markdown(
            f"""Interpretation: CKD is close to **linearly separable** in this dataset. A single threshold on hemoglobin alone
classifies ~91% of patients correctly, a plain logistic regression reaches CV ROC-AUC ≈ 1.00, and the healthy controls were mostly young, normotensive,
non-diabetic people with textbook lab values. Near-perfect accuracy is therefore expected and is **not** a sign of leakage,
but it also means the task is much easier than real screening, where early-stage CKD overlaps heavily with normal.
Note also a collection artefact: {miss['positive']:.0%} of CKD rows have at least one missing value vs {miss['negative']:.0%} of
non-CKD rows, so "missingness" itself carries signal.""")

    st.markdown("#### Limitations")
    lim = {
        "ckd": """- 400 patients from one hospital in India over two months; controls are unusually healthy, so the model
has never seen the hard cases (stage 1–2 CKD, other kidney conditions).
- The test set has only 80 patients: one extra error changes accuracy by 1.25 points.
- Reference ranges shown in the form are generic adult values and differ between labs.""",
        "heart": """- 303 patients from 1988 (Cleveland Clinic), all referred for angiography, so prevalence (~46%) is far higher
than in the general population and the probabilities are not calibrated for screening.
- Several inputs (thallium scan, fluoroscopy vessel count) come from tests that are only done after a referral.
- Sex is 68% male; performance for women is less certain. Test set: 61 patients, so metrics have wide uncertainty.""",
    }[task]
    st.markdown(lim + "\n- Probabilities come from a model trained on historical data and have not been clinically validated.")


def about_tab():
    st.markdown("""
#### How it works
1. **Data**: two public UCI datasets, downloaded by `scripts/download_data.py`; exact duplicate rows are removed **before** splitting.
2. **Split**: 80% train / 20% held-out test, stratified by outcome. The test set is used once, at the end.
3. **Pipeline** (`crp/pipeline.py`): median / most-frequent imputation → one-hot encoding → standard scaling → model,
   all inside one sklearn `Pipeline`, so nothing is fitted on validation or test data.
4. **Model comparison**: Logistic Regression, Decision Tree, Random Forest and XGBoost, scored with stratified 5-fold CV
   (accuracy, F1, ROC-AUC, mean ± std). The best mean ROC-AUC is deployed.
5. **Feature selection (CKD)**: permutation importance (log-loss) averaged over CV folds, 24 → 12 features.
6. **Explanations**: SHAP (TreeExplainer / LinearExplainer). One-hot columns are summed back into their original feature so
   every factor reads as one clinical variable.
""")
    st.graphviz_chart("""
digraph { rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#CCFBF1", color="#0F766E", fontname="Inter", fontsize=11];
  edge [color="#0F766E"];
  A [label="UCI download"]; B [label="clean + de-duplicate"]; C [label="stratified 80/20 split"];
  D [label="impute → encode → scale → model"]; E [label="5-fold CV\\nLR · DT · RF · XGB"];
  F [label="best CV ROC-AUC\\n→ held-out test"]; G [label="SHAP"]; H [label="Streamlit UI"];
  A -> B -> C -> D -> E -> F -> G -> H;
}""", use_container_width=True)
    st.markdown("Built by **Amna Choudhri**. Source code, metrics and figures are in the repository README.")


tabs = st.tabs(["🫘  Kidney (CKD)", "❤️  Heart disease", "📊  Model card", "ℹ️  About / How it works"])
with tabs[0]:
    predict_tab("ckd")
with tabs[1]:
    predict_tab("heart")
with tabs[2]:
    model_card()
with tabs[3]:
    about_tab()
