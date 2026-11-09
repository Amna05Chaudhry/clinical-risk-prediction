"""Feature metadata shared by the training pipeline and the UI.

Every feature has a human label, a unit, a widget type and (where it makes sense)
a typical adult reference range. Reference ranges are indicative only and vary
between labs; they are used to phrase explanations, not to make decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    kind: str  # "num" | "cat"
    unit: str = ""
    normal: tuple[float, float] | None = None  # reference range (numeric)
    options: dict = field(default_factory=dict)  # stored value -> display text (categorical)
    help: str = ""
    step: float = 1.0
    ordinal: bool = False  # numeric-coded categorical kept as a number (e.g. 0-5 grade)


# ------------------------------------------------------------------ CKD
CKD_FEATURES: list[Feature] = [
    Feature("age", "Age", "num", "years", None, step=1),
    Feature("bp", "Blood pressure (diastolic)", "num", "mm Hg", (60, 80), step=5),
    Feature("sg", "Urine specific gravity", "num", "", (1.015, 1.025), step=0.005,
            help="Concentration of urine. Low values suggest the kidneys cannot concentrate urine."),
    Feature("al", "Urine albumin", "num", "grade 0-5", (0, 0), step=1, ordinal=True,
            help="Dipstick protein grade. Any albumin in urine is a classic kidney-damage sign."),
    Feature("su", "Urine sugar", "num", "grade 0-5", (0, 0), step=1, ordinal=True),
    Feature("rbc", "Red blood cells in urine", "cat", options={"normal": "Normal", "abnormal": "Abnormal"}),
    Feature("pc", "Pus cells in urine", "cat", options={"normal": "Normal", "abnormal": "Abnormal"}),
    Feature("pcc", "Pus cell clumps", "cat", options={"notpresent": "Not present", "present": "Present"}),
    Feature("ba", "Bacteria in urine", "cat", options={"notpresent": "Not present", "present": "Present"}),
    Feature("bgr", "Random blood glucose", "num", "mg/dL", (70, 140), step=1),
    Feature("bu", "Blood urea", "num", "mg/dL", (15, 40), step=1),
    Feature("sc", "Serum creatinine", "num", "mg/dL", (0.6, 1.2), step=0.1,
            help="Waste product filtered by the kidneys; rises when kidney function drops."),
    Feature("sod", "Sodium", "num", "mEq/L", (135, 145), step=1),
    Feature("pot", "Potassium", "num", "mEq/L", (3.5, 5.0), step=0.1),
    Feature("hemo", "Hemoglobin", "num", "g/dL", (12, 17), step=0.1,
            help="CKD often causes anaemia because damaged kidneys make less erythropoietin."),
    Feature("pcv", "Packed cell volume (haematocrit)", "num", "%", (36, 50), step=1),
    Feature("wbcc", "White blood cell count", "num", "cells/µL", (4000, 11000), step=100),
    Feature("rbcc", "Red blood cell count", "num", "million/µL", (4.2, 5.9), step=0.1),
    Feature("htn", "Hypertension", "cat", options={"no": "No", "yes": "Yes"}),
    Feature("dm", "Diabetes mellitus", "cat", options={"no": "No", "yes": "Yes"}),
    Feature("cad", "Coronary artery disease", "cat", options={"no": "No", "yes": "Yes"}),
    Feature("appet", "Appetite", "cat", options={"good": "Good", "poor": "Poor"}),
    Feature("pe", "Pedal (ankle) oedema", "cat", options={"no": "No", "yes": "Yes"}),
    Feature("ane", "Anaemia", "cat", options={"no": "No", "yes": "Yes"}),
]

# ------------------------------------------------------------------ Heart
HEART_FEATURES: list[Feature] = [
    Feature("age", "Age", "num", "years", None, step=1),
    Feature("sex", "Sex", "cat", options={1: "Male", 0: "Female"}),
    Feature("cp", "Chest pain type", "cat",
            options={1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "No symptoms (asymptomatic)"},
            help="In this dataset, 'asymptomatic' patients referred for angiography are the highest-risk group."),
    Feature("trestbps", "Resting blood pressure (systolic)", "num", "mm Hg", (90, 120), step=1),
    Feature("chol", "Serum cholesterol", "num", "mg/dL", (125, 200), step=1),
    Feature("fbs", "Fasting blood sugar > 120 mg/dL", "cat", options={0: "No", 1: "Yes"}),
    Feature("restecg", "Resting ECG", "cat",
            options={0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}),
    Feature("thalach", "Max heart rate on exercise test", "num", "bpm", None, step=1,
            help="Lower peak heart rate on a stress test is associated with heart disease."),
    Feature("exang", "Exercise-induced angina", "cat", options={0: "No", 1: "Yes"}),
    Feature("oldpeak", "ST depression on exercise", "num", "mm", (0, 1), step=0.1,
            help="ST-segment depression induced by exercise relative to rest."),
    Feature("slope", "ST segment slope at peak exercise", "cat", options={1: "Upsloping", 2: "Flat", 3: "Downsloping"}),
    Feature("ca", "Major vessels coloured by fluoroscopy", "num", "0-3", (0, 0), step=1, ordinal=True),
    Feature("thal", "Thallium stress test", "cat", options={3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}),
]

FEATURES = {"ckd": CKD_FEATURES, "heart": HEART_FEATURES}


def feature_map(task: str) -> dict[str, Feature]:
    return {f.name: f for f in FEATURES[task]}
