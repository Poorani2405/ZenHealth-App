"""
=============================================================
  AI Health & Nutrition Advisor — FastAPI Prediction API
=============================================================
Endpoints:
  GET  /                        → Health check
  GET  /models                  → List all loaded models & scores
  POST /predict/obesity-risk    → Predict obesity risk level
  POST /predict/diabetes-risk   → Predict diabetes risk level
  POST /predict/hypertension-risk → Predict hypertension risk level
  POST /predict/bmi-category    → Predict BMI category
  POST /predict/badge-status    → Predict achievement badge
  POST /predict/calorie-target  → Predict daily calorie target
  POST /predict/all             → Run ALL 6 predictions at once

Run:
  pip install fastapi uvicorn
  uvicorn app:app --reload --port 8000

Docs (auto-generated):
  http://localhost:8000/docs
"""

import os, json, pickle, math
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "models"

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL MODELS AT STARTUP
# ══════════════════════════════════════════════════════════════════════════════
print("\n🔄  Loading models...")

TASKS = [
    "obesity_risk",
    "diabetes_risk",
    "hypertension_risk",
    "bmi_category",
    "badge_status",
    "calorie_target",
]

models    = {}   # task → sklearn model
scalers   = {}   # task → StandardScaler
metas     = {}   # task → dict (features, task_type, best_model, score)
label_enc = {}   # column → LabelEncoder

# Load label encoders
le_path = MODEL_DIR / "label_encoders.pkl"
if le_path.exists():
    with open(le_path, "rb") as f:
        label_enc = pickle.load(f)
    print(f"  ✓ Label encoders loaded ({len(label_enc)} columns)")
else:
    raise FileNotFoundError(f"label_encoders.pkl not found in {MODEL_DIR}")

# Load each task's model + scaler + meta
for task in TASKS:
    model_path  = MODEL_DIR / f"{task}_best_model.pkl"
    scaler_path = MODEL_DIR / f"{task}_scaler.pkl"
    meta_path   = MODEL_DIR / f"{task}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path,  "rb") as f: models[task]  = pickle.load(f)
    with open(scaler_path, "rb") as f: scalers[task] = pickle.load(f)
    with open(meta_path)         as f: metas[task]   = json.load(f)
    print(f"  ✓ {task:30s} → {metas[task]['best_model']}  (score={metas[task]['best_score']})")

print("✅  All models ready.\n")

# ── Encode helper ─────────────────────────────────────────────────────────────
BUDGET_MAP = {
    "low":     1500,
    "medium":  3500,
    "high":    7500,
    "premium": 12000,
}

def encode_value(col: str, val):
    """Encode a categorical value using the saved LabelEncoder."""
    if col in label_enc:
        le = label_enc[col]
        try:
            return int(le.transform([str(val)])[0])
        except ValueError:
            # unseen label → use 0 (most common fallback)
            return 0
    return val

def build_feature_vector(task: str, data: dict) -> np.ndarray:
    """Build a numpy feature vector in the exact order the model expects."""
    feature_names = metas[task]["features"]
    vec = []
    for feat in feature_names:
        raw = data.get(feat, 0)
        # Auto-encode if the column is categorical
        if feat in label_enc:
            val = encode_value(feat, raw)
        else:
            val = float(raw) if raw is not None else 0.0
        vec.append(val)
    return np.array([vec], dtype=float)

def decode_prediction(task: str, encoded_pred) -> str:
    """Decode an encoded integer prediction back to its original label."""
    target = metas[task]["target"]
    if target in label_enc:
        return str(label_enc[target].inverse_transform([int(encoded_pred)])[0])
    return str(encoded_pred)

def run_prediction(task: str, feature_vec: np.ndarray):
    """Run model inference and return prediction + probabilities."""
    model  = models[task]
    scaler = scalers[task]

    # Tree-based models don't need scaling; linear/SVM do
    model_name = metas[task]["best_model"].lower()
    needs_scale = any(k in model_name for k in ["logistic", "ridge", "svm", "knn", "k-nearest"])
    X = scaler.transform(feature_vec) if needs_scale else feature_vec

    pred = model.predict(X)[0]

    proba_dict = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        target = metas[task]["target"]
        if target in label_enc:
            classes = label_enc[target].classes_
        else:
            classes = [str(c) for c in model.classes_]
        proba_dict = {str(cls): round(float(p), 4) for cls, p in zip(classes, proba)}

    task_type = metas[task]["task_type"]
    if task_type == "regression":
        return round(float(pred), 2), None
    else:
        return decode_prediction(task, pred), proba_dict


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class UserProfile(BaseModel):
    """Core user profile — used by most prediction endpoints."""

    # Demographics
    age:            int   = Field(..., ge=1,   le=120,  example=35,    description="Age in years")
    gender:         str   = Field(...,                   example="Male", description="Male / Female / Other")
    height_cm:      float = Field(..., ge=100, le=250,  example=170.0, description="Height in centimeters")
    weight_kg:      float = Field(..., ge=20,  le=300,  example=75.0,  description="Weight in kilograms")

    # Lifestyle
    activity_level: str   = Field(..., example="Moderately Active",
                                  description="Sedentary / Lightly Active / Moderately Active / Very Active / Extremely Active")
    dietary_preference: str = Field(..., example="Vegetarian",
                                    description="Vegetarian / Vegan / Non-Vegetarian / Keto / Paleo / Mediterranean / Gluten-Free / Dairy-Free")
    health_goal:    str   = Field(..., example="Weight Loss",
                                  description="Weight Loss / Muscle Gain / Maintain Weight / General Health / Improve Stamina / Reduce Stress / Better Sleep")
    food_allergy:   str   = Field(default="None", example="None",
                                  description="None / Dairy / Gluten / Nuts / Shellfish / Eggs / Soy / Peanuts / Wheat / Fish")
    medical_history: str  = Field(default="None", example="None",
                                  description="None / Diabetes Type 2 / Hypertension / PCOD / Thyroid / Asthma / Anemia / High Cholesterol")

    # Health metrics
    avg_sleep_hours:       float = Field(..., ge=0,  le=24,    example=7.0,  description="Average sleep hours per night")
    daily_steps:           int   = Field(..., ge=0,  le=50000, example=8000, description="Average daily steps")
    water_intake_litres:   float = Field(..., ge=0,  le=10,    example=2.5,  description="Daily water intake in litres")
    calories_burned_per_day: int = Field(..., ge=0,  le=10000, example=2200, description="Estimated calories burned per day")
    avg_heart_rate_bpm:    int   = Field(..., ge=30, le=220,   example=72,   description="Average resting heart rate (BPM)")
    stress_level:          str   = Field(..., example="Medium", description="Low / Medium / High")
    sleep_quality_score:   str   = Field(..., example="Good",   description="Poor / Fair / Good / Excellent")

    # Fitness
    fitness_level:         str   = Field(..., example="Intermediate", description="Beginner / Intermediate / Advanced")
    workout_days_per_week: int   = Field(..., ge=0, le=7, example=4,  description="Workout days per week (0–7)")

    # Nutrition
    meal_frequency_per_day: int  = Field(..., ge=1, le=10, example=3, description="Number of meals per day")
    daily_calorie_target:   int  = Field(default=2000, ge=500, le=6000, example=2000)
    protein_target_g:       int  = Field(default=100,  ge=0,   le=500,  example=100)
    carbs_target_g:         int  = Field(default=250,  ge=0,   le=800,  example=250)
    fat_target_g:           int  = Field(default=65,   ge=0,   le=300,  example=65)

    # Tracking
    current_streak_days:   int   = Field(default=0,  ge=0,  le=1000, example=15)
    total_workouts_logged: int   = Field(default=0,  ge=0,  le=5000, example=50)
    total_meals_logged:    int   = Field(default=0,  ge=0,  le=5000, example=120)
    number_of_people:      int   = Field(default=1,  ge=1,  le=20,   example=2)
    budget_range:          str   = Field(default="medium",
                                         example="medium",
                                         description="low / medium / high / premium")

    # Auto-computed if not supplied
    bmr_kcal:  Optional[float] = Field(default=None, description="Basal Metabolic Rate (auto-calculated if omitted)")
    tdee_kcal: Optional[float] = Field(default=None, description="Total Daily Energy Expenditure (auto-calculated if omitted)")
    bmi:       Optional[float] = Field(default=None, description="BMI (auto-calculated if omitted)")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        allowed = ["Male", "Female", "Other"]
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v

    @field_validator("stress_level")
    @classmethod
    def validate_stress(cls, v):
        allowed = ["Low", "Medium", "High"]
        if v not in allowed:
            raise ValueError(f"stress_level must be one of {allowed}")
        return v

    @field_validator("sleep_quality_score")
    @classmethod
    def validate_sleep_quality(cls, v):
        allowed = ["Poor", "Fair", "Good", "Excellent"]
        if v not in allowed:
            raise ValueError(f"sleep_quality_score must be one of {allowed}")
        return v

    @field_validator("fitness_level")
    @classmethod
    def validate_fitness(cls, v):
        allowed = ["Beginner", "Intermediate", "Advanced"]
        if v not in allowed:
            raise ValueError(f"fitness_level must be one of {allowed}")
        return v

    def compute_derived(self) -> dict:
        """Auto-compute BMI, BMR, TDEE if not provided, return as flat dict."""
        d = self.model_dump()

        # BMI
        if d["bmi"] is None:
            d["bmi"] = round(d["weight_kg"] / ((d["height_cm"] / 100) ** 2), 1)

        # BMR (Mifflin-St Jeor)
        if d["bmr_kcal"] is None:
            w, h, a = d["weight_kg"], d["height_cm"], d["age"]
            if d["gender"] == "Male":
                d["bmr_kcal"] = round(10*w + 6.25*h - 5*a + 5, 1)
            elif d["gender"] == "Female":
                d["bmr_kcal"] = round(10*w + 6.25*h - 5*a - 161, 1)
            else:
                d["bmr_kcal"] = round(10*w + 6.25*h - 5*a - 78, 1)

        # TDEE
        if d["tdee_kcal"] is None:
            multipliers = {
                "Sedentary": 1.2, "Lightly Active": 1.375,
                "Moderately Active": 1.55, "Very Active": 1.725, "Extremely Active": 1.9
            }
            mult = multipliers.get(d["activity_level"], 1.55)
            d["tdee_kcal"] = round(d["bmr_kcal"] * mult)

        # Budget numeric
        d["budget_numeric"] = BUDGET_MAP.get(d["budget_range"].lower(), 3500)

        return d


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Health & Nutrition Advisor API",
    description=(
        "Prediction API for the AI-based Personalized Nutrition and Health Advisor App.\n\n"
        "Submit user health data and receive ML-powered predictions for:\n"
        "- Health risk levels (obesity, diabetes, hypertension)\n"
        "- BMI category classification\n"
        "- Daily calorie target estimation\n"
        "- Achievement badge status\n\n"
        "All 6 predictions can be retrieved in a single call via `/predict/all`."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Status"])
def root():
    return {
        "status": "running",
        "api": "AI Health & Nutrition Advisor",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [f"/predict/{t.replace('_','-')}" for t in TASKS] + ["/predict/all"],
    }

@app.get("/models", tags=["Status"])
def list_models():
    """Returns all loaded models with their algorithm and accuracy score."""
    return {
        task: {
            "algorithm":  metas[task]["best_model"],
            "score":      metas[task]["best_score"],
            "task_type":  metas[task]["task_type"],
            "n_features": len(metas[task]["features"]),
        }
        for task in TASKS
    }

# ── Individual prediction endpoints ───────────────────────────────────────────

@app.post("/predict/obesity-risk", tags=["Predictions"])
def predict_obesity_risk(user: UserProfile):
    """
    Predict obesity risk level: **Low / Moderate / High**

    Uses: BMI, activity level, sleep, steps, medical history, diet, stress.
    """
    data = user.compute_derived()
    vec  = build_feature_vector("obesity_risk", data)
    pred, proba = run_prediction("obesity_risk", vec)
    return {
        "prediction":    pred,
        "probabilities": proba,
        "model_used":    metas["obesity_risk"]["best_model"],
        "disclaimer":    "This is a screening tool, not a medical diagnosis.",
    }

@app.post("/predict/diabetes-risk", tags=["Predictions"])
def predict_diabetes_risk(user: UserProfile):
    """
    Predict diabetes risk level: **Low / Moderate / High**

    Uses: BMI, medical history, dietary preference, activity, calorie data.
    """
    data = user.compute_derived()
    vec  = build_feature_vector("diabetes_risk", data)
    pred, proba = run_prediction("diabetes_risk", vec)
    return {
        "prediction":    pred,
        "probabilities": proba,
        "model_used":    metas["diabetes_risk"]["best_model"],
        "disclaimer":    "This is a screening tool, not a medical diagnosis.",
    }

@app.post("/predict/hypertension-risk", tags=["Predictions"])
def predict_hypertension_risk(user: UserProfile):
    """
    Predict hypertension risk level: **Low / Moderate / High**

    Uses: stress level, activity, BMI, sleep, medical history.
    """
    data = user.compute_derived()
    vec  = build_feature_vector("hypertension_risk", data)
    pred, proba = run_prediction("hypertension_risk", vec)
    return {
        "prediction":    pred,
        "probabilities": proba,
        "model_used":    metas["hypertension_risk"]["best_model"],
        "disclaimer":    "This is a screening tool, not a medical diagnosis.",
    }

@app.post("/predict/bmi-category", tags=["Predictions"])
def predict_bmi_category(user: UserProfile):
    """
    Predict BMI category: **Underweight / Normal / Overweight / Obese**

    BMI is auto-calculated from height and weight if not provided.
    """
    data = user.compute_derived()
    vec  = build_feature_vector("bmi_category", data)
    pred, proba = run_prediction("bmi_category", vec)
    return {
        "prediction":  pred,
        "bmi_value":   data["bmi"],
        "probabilities": proba,
        "model_used":  metas["bmi_category"]["best_model"],
    }

@app.post("/predict/badge-status", tags=["Predictions"])
def predict_badge_status(user: UserProfile):
    """
    Predict achievement badge: **Beginner / Bronze / Silver / Gold / Platinum**

    Based on streak, workout logs, meals logged, and engagement.
    """
    data = user.compute_derived()
    vec  = build_feature_vector("badge_status", data)
    pred, proba = run_prediction("badge_status", vec)
    return {
        "prediction":    pred,
        "probabilities": proba,
        "model_used":    metas["badge_status"]["best_model"],
    }

@app.post("/predict/calorie-target", tags=["Predictions"])
def predict_calorie_target(user: UserProfile):
    """
    Predict personalized daily calorie target (kcal).

    TDEE and BMR are auto-calculated from height, weight, age, and activity level.
    """
    data = user.compute_derived()
    vec  = build_feature_vector("calorie_target", data)
    pred, _ = run_prediction("calorie_target", vec)

    # Macro split based on goal
    goal = data["health_goal"]
    if goal == "Muscle Gain":
        p_pct, c_pct, f_pct = 0.30, 0.45, 0.25
    elif goal == "Weight Loss":
        p_pct, c_pct, f_pct = 0.35, 0.35, 0.30
    else:
        p_pct, c_pct, f_pct = 0.25, 0.50, 0.25

    cal = float(pred)
    return {
        "daily_calorie_target_kcal": pred,
        "bmr_kcal":                  data["bmr_kcal"],
        "tdee_kcal":                 data["tdee_kcal"],
        "macros": {
            "protein_g":      round((cal * p_pct) / 4),
            "carbohydrates_g": round((cal * c_pct) / 4),
            "fat_g":           round((cal * f_pct) / 9),
        },
        "model_used": metas["calorie_target"]["best_model"],
        "r2_score":   metas["calorie_target"]["best_score"],
    }

# ── Master endpoint — ALL predictions at once ─────────────────────────────────

@app.post("/predict/all", tags=["Predictions"])
def predict_all(user: UserProfile):
    """
    Run **all 6 predictions** in one request.

    Returns obesity risk, diabetes risk, hypertension risk, BMI category,
    badge status, and calorie target — plus derived health stats.
    """
    data = user.compute_derived()
    results = {}

    for task in TASKS:
        vec  = build_feature_vector(task, data)
        pred, proba = run_prediction(task, vec)
        results[task] = {
            "prediction":  pred,
            "probabilities": proba,
            "model":       metas[task]["best_model"],
        }

    # Calorie macro breakdown
    goal = data["health_goal"]
    cal  = float(results["calorie_target"]["prediction"])
    if goal == "Muscle Gain":
        p_pct, c_pct, f_pct = 0.30, 0.45, 0.25
    elif goal == "Weight Loss":
        p_pct, c_pct, f_pct = 0.35, 0.35, 0.30
    else:
        p_pct, c_pct, f_pct = 0.25, 0.50, 0.25

    return {
        "user_stats": {
            "bmi":       data["bmi"],
            "bmr_kcal":  data["bmr_kcal"],
            "tdee_kcal": data["tdee_kcal"],
        },
        "predictions": {
            "obesity_risk":       results["obesity_risk"]["prediction"],
            "diabetes_risk":      results["diabetes_risk"]["prediction"],
            "hypertension_risk":  results["hypertension_risk"]["prediction"],
            "bmi_category":       results["bmi_category"]["prediction"],
            "badge_status":       results["badge_status"]["prediction"],
            "daily_calorie_target_kcal": results["calorie_target"]["prediction"],
            "macros": {
                "protein_g":       round((cal * p_pct) / 4),
                "carbohydrates_g": round((cal * c_pct) / 4),
                "fat_g":           round((cal * f_pct) / 9),
            },
        },
        "probabilities": {
            task: results[task]["probabilities"]
            for task in TASKS if results[task]["probabilities"]
        },
        "disclaimer": "Predictions are AI-generated and not a substitute for professional medical advice.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)