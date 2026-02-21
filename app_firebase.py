"""
=============================================================
  AI Health & Nutrition Advisor — FastAPI + Firebase API
=============================================================
- Saves every prediction to Firebase Firestore
- Reads user profile from Firebase
- Works with Android/iOS APK over same WiFi or internet

Setup:
  1. pip install fastapi uvicorn firebase-admin
  2. Download serviceAccountKey.json from Firebase Console
  3. python -m uvicorn app_firebase:app --host 0.0.0.0 --port 8000 --reload
=============================================================
"""

import os, json, pickle, math
from pathlib import Path
from typing import Optional
from datetime import datetime
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

# ── Firebase Setup ─────────────────────────────────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    SERVICE_KEY = Path(__file__).parent / "serviceAccountKey.json"
    if SERVICE_KEY.exists():
        cred = credentials.Certificate(str(SERVICE_KEY))
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        FIREBASE_ENABLED = True
        print("✅  Firebase connected!")
    else:
        FIREBASE_ENABLED = False
        db = None
        print("⚠️   Firebase disabled — serviceAccountKey.json not found")
        print("     API works normally, predictions won't be saved to DB")

except ImportError:
    FIREBASE_ENABLED = False
    db = None
    print("⚠️   firebase-admin not installed — run: pip install firebase-admin")
    print("     API works normally without Firebase")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n🔄  Loading models...")

TASKS = [
    "obesity_risk", "diabetes_risk", "hypertension_risk",
    "bmi_category", "badge_status", "calorie_target",
]

models    = {}
scalers   = {}
metas     = {}
label_enc = {}

with open(MODEL_DIR / "label_encoders.pkl", "rb") as f:
    label_enc = pickle.load(f)

for task in TASKS:
    with open(MODEL_DIR / f"{task}_best_model.pkl", "rb") as f: models[task]  = pickle.load(f)
    with open(MODEL_DIR / f"{task}_scaler.pkl",     "rb") as f: scalers[task] = pickle.load(f)
    with open(MODEL_DIR / f"{task}_meta.json")           as f: metas[task]   = json.load(f)
    print(f"  ✓ {task}")

print("✅  All models ready.\n")

# ── Helpers ────────────────────────────────────────────────────────────────────
BUDGET_MAP = {"low": 1500, "medium": 3500, "high": 7500, "premium": 12000}

def encode_value(col, val):
    if col in label_enc:
        try:    return int(label_enc[col].transform([str(val)])[0])
        except: return 0
    return val

def build_feature_vector(task, data):
    vec = []
    for feat in metas[task]["features"]:
        raw = data.get(feat, 0)
        val = encode_value(feat, raw) if feat in label_enc else float(raw or 0)
        vec.append(val)
    return np.array([vec], dtype=float)

def decode_prediction(task, pred):
    target = metas[task]["target"]
    if target in label_enc:
        return str(label_enc[target].inverse_transform([int(pred)])[0])
    return str(pred)

def run_prediction(task, vec):
    model      = models[task]
    model_name = metas[task]["best_model"].lower()
    needs_sc   = any(k in model_name for k in ["logistic", "ridge", "svm", "knn"])
    X          = scalers[task].transform(vec) if needs_sc else vec
    pred       = model.predict(X)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        p      = model.predict_proba(X)[0]
        target = metas[task]["target"]
        cls    = label_enc[target].classes_ if target in label_enc else model.classes_
        proba  = {str(c): round(float(v), 4) for c, v in zip(cls, p)}

    if metas[task]["task_type"] == "regression":
        return round(float(pred), 2), None
    return decode_prediction(task, pred), proba

def save_to_firebase(collection, data):
    """Save a document to Firestore. Silently skips if Firebase is disabled."""
    if not FIREBASE_ENABLED or db is None:
        return None
    try:
        ref = db.collection(collection).add({**data, "timestamp": datetime.utcnow()})
        return ref[1].id
    except Exception as e:
        print(f"  Firebase write error: {e}")
        return None

def get_from_firebase(collection, doc_id):
    """Fetch a document from Firestore."""
    if not FIREBASE_ENABLED or db is None:
        return None
    try:
        doc = db.collection(collection).document(doc_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print(f"  Firebase read error: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class UserProfile(BaseModel):
    # Required
    user_id:                 Optional[str]   = Field(default=None,  description="Firebase user UID (optional)")
    age:                     int             = Field(..., ge=1,   le=120,  example=28)
    gender:                  str             = Field(...,                   example="Female")
    height_cm:               float           = Field(..., ge=100, le=250,  example=162.0)
    weight_kg:               float           = Field(..., ge=20,  le=300,  example=68.0)
    activity_level:          str             = Field(..., example="Moderately Active")
    dietary_preference:      str             = Field(..., example="Vegetarian")
    health_goal:             str             = Field(..., example="Weight Loss")
    avg_sleep_hours:         float           = Field(..., ge=0,  le=24,    example=6.5)
    daily_steps:             int             = Field(..., ge=0,  le=50000, example=7200)
    water_intake_litres:     float           = Field(..., ge=0,  le=10,    example=2.2)
    calories_burned_per_day: int             = Field(..., ge=0,  le=10000, example=1950)
    avg_heart_rate_bpm:      int             = Field(..., ge=30, le=220,   example=76)
    stress_level:            str             = Field(..., example="Medium")
    sleep_quality_score:     str             = Field(..., example="Fair")
    fitness_level:           str             = Field(..., example="Beginner")
    workout_days_per_week:   int             = Field(..., ge=0, le=7,      example=3)
    meal_frequency_per_day:  int             = Field(..., ge=1, le=10,     example=4)

    # Optional with defaults
    food_allergy:            str             = Field(default="None")
    medical_history:         str             = Field(default="None")
    daily_calorie_target:    int             = Field(default=2000)
    protein_target_g:        int             = Field(default=100)
    carbs_target_g:          int             = Field(default=250)
    fat_target_g:            int             = Field(default=65)
    current_streak_days:     int             = Field(default=0)
    total_workouts_logged:   int             = Field(default=0)
    total_meals_logged:      int             = Field(default=0)
    number_of_people:        int             = Field(default=1)
    budget_range:            str             = Field(default="medium")
    bmr_kcal:                Optional[float] = None
    tdee_kcal:               Optional[float] = None
    bmi:                     Optional[float] = None

    def compute_derived(self):
        d = self.model_dump()
        if d["bmi"] is None:
            d["bmi"] = round(d["weight_kg"] / ((d["height_cm"] / 100) ** 2), 1)
        if d["bmr_kcal"] is None:
            w, h, a = d["weight_kg"], d["height_cm"], d["age"]
            if d["gender"] == "Male":
                d["bmr_kcal"] = round(10*w + 6.25*h - 5*a + 5, 1)
            elif d["gender"] == "Female":
                d["bmr_kcal"] = round(10*w + 6.25*h - 5*a - 161, 1)
            else:
                d["bmr_kcal"] = round(10*w + 6.25*h - 5*a - 78, 1)
        if d["tdee_kcal"] is None:
            m = {"Sedentary":1.2,"Lightly Active":1.375,"Moderately Active":1.55,"Very Active":1.725,"Extremely Active":1.9}
            d["tdee_kcal"] = round(d["bmr_kcal"] * m.get(d["activity_level"], 1.55))
        d["budget_numeric"] = BUDGET_MAP.get(d["budget_range"].lower(), 3500)
        return d

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Health & Nutrition Advisor API",
    description="FastAPI + Firebase ML prediction API for health & nutrition app.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
def root():
    return {
        "status":           "running",
        "version":          "2.0.0",
        "firebase":         FIREBASE_ENABLED,
        "models_loaded":    len(models),
        "docs":             "/docs",
    }

@app.get("/models", tags=["Status"])
def list_models():
    return {
        task: {
            "algorithm": metas[task]["best_model"],
            "score":     metas[task]["best_score"],
            "type":      metas[task]["task_type"],
        }
        for task in TASKS
    }

# ── Firebase User Endpoints ───────────────────────────────────────────────────

@app.post("/user/save", tags=["Firebase"])
def save_user_profile(user: UserProfile):
    """
    Save a user profile to Firebase Firestore.
    Pass user_id = Firebase Auth UID to link to the authenticated user.
    """
    data      = user.compute_derived()
    uid       = data.get("user_id") or "anonymous"
    doc_id    = save_to_firebase(f"users/{uid}/profile", data)
    return {
        "saved":     FIREBASE_ENABLED,
        "user_id":   uid,
        "doc_id":    doc_id,
        "message":   "Profile saved to Firebase" if FIREBASE_ENABLED else "Firebase not connected",
    }

@app.get("/user/{user_id}/history", tags=["Firebase"])
def get_prediction_history(user_id: str):
    """
    Fetch all past predictions for a user from Firebase.
    """
    if not FIREBASE_ENABLED:
        raise HTTPException(status_code=503, detail="Firebase not connected")
    try:
        docs = db.collection("predictions").where("user_id", "==", user_id).stream()
        history = [{"id": d.id, **d.to_dict()} for d in docs]
        return {"user_id": user_id, "count": len(history), "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Prediction Endpoints ───────────────────────────────────────────────────────

@app.post("/predict/all", tags=["Predictions"])
def predict_all(user: UserProfile):
    """
    Run all 6 predictions and save results to Firebase automatically.
    """
    data    = user.compute_derived()
    results = {}

    for task in TASKS:
        vec              = build_feature_vector(task, data)
        pred, proba      = run_prediction(task, vec)
        results[task]    = {"prediction": pred, "probabilities": proba}

    cal  = float(results["calorie_target"]["prediction"])
    goal = data["health_goal"]
    if goal == "Muscle Gain":      p, c, f = 0.30, 0.45, 0.25
    elif goal == "Weight Loss":    p, c, f = 0.35, 0.35, 0.30
    else:                          p, c, f = 0.25, 0.50, 0.25

    predictions = {
        "obesity_risk":              results["obesity_risk"]["prediction"],
        "diabetes_risk":             results["diabetes_risk"]["prediction"],
        "hypertension_risk":         results["hypertension_risk"]["prediction"],
        "bmi_category":              results["bmi_category"]["prediction"],
        "badge_status":              results["badge_status"]["prediction"],
        "daily_calorie_target_kcal": results["calorie_target"]["prediction"],
        "macros": {
            "protein_g":        round((cal * p) / 4),
            "carbohydrates_g":  round((cal * c) / 4),
            "fat_g":            round((cal * f) / 9),
        },
    }

    user_stats = {
        "bmi":       data["bmi"],
        "bmr_kcal":  data["bmr_kcal"],
        "tdee_kcal": data["tdee_kcal"],
    }

    # ── Save to Firebase ──────────────────────────────────────────────────────
    uid    = data.get("user_id") or "anonymous"
    doc_id = save_to_firebase("predictions", {
        "user_id":    uid,
        "user_stats": user_stats,
        "predictions": predictions,
        "input":      {k: v for k, v in data.items() if k not in ["bmr_kcal","tdee_kcal","bmi","budget_numeric"]},
    })

    return {
        "user_stats":  user_stats,
        "predictions": predictions,
        "probabilities": {
            task: results[task]["probabilities"]
            for task in TASKS if results[task]["probabilities"]
        },
        "firebase": {
            "saved":  FIREBASE_ENABLED,
            "doc_id": doc_id,
        },
        "disclaimer": "AI predictions — not a substitute for medical advice.",
    }

@app.post("/predict/obesity-risk",      tags=["Predictions"])
def predict_obesity(user: UserProfile):
    data = user.compute_derived()
    pred, proba = run_prediction("obesity_risk", build_feature_vector("obesity_risk", data))
    save_to_firebase("predictions", {"user_id": data.get("user_id"), "task": "obesity_risk", "prediction": pred})
    return {"prediction": pred, "probabilities": proba}

@app.post("/predict/diabetes-risk",     tags=["Predictions"])
def predict_diabetes(user: UserProfile):
    data = user.compute_derived()
    pred, proba = run_prediction("diabetes_risk", build_feature_vector("diabetes_risk", data))
    save_to_firebase("predictions", {"user_id": data.get("user_id"), "task": "diabetes_risk", "prediction": pred})
    return {"prediction": pred, "probabilities": proba}

@app.post("/predict/hypertension-risk", tags=["Predictions"])
def predict_hypertension(user: UserProfile):
    data = user.compute_derived()
    pred, proba = run_prediction("hypertension_risk", build_feature_vector("hypertension_risk", data))
    save_to_firebase("predictions", {"user_id": data.get("user_id"), "task": "hypertension_risk", "prediction": pred})
    return {"prediction": pred, "probabilities": proba}

@app.post("/predict/bmi-category",      tags=["Predictions"])
def predict_bmi(user: UserProfile):
    data = user.compute_derived()
    pred, proba = run_prediction("bmi_category", build_feature_vector("bmi_category", data))
    return {"prediction": pred, "bmi_value": data["bmi"], "probabilities": proba}

@app.post("/predict/badge-status",      tags=["Predictions"])
def predict_badge(user: UserProfile):
    data = user.compute_derived()
    pred, proba = run_prediction("badge_status", build_feature_vector("badge_status", data))
    return {"prediction": pred, "probabilities": proba}

@app.post("/predict/calorie-target",    tags=["Predictions"])
def predict_calories(user: UserProfile):
    data = user.compute_derived()
    pred, _ = run_prediction("calorie_target", build_feature_vector("calorie_target", data))
    cal  = float(pred)
    goal = data["health_goal"]
    if goal == "Muscle Gain":   p, c, f = 0.30, 0.45, 0.25
    elif goal == "Weight Loss": p, c, f = 0.35, 0.35, 0.30
    else:                       p, c, f = 0.25, 0.50, 0.25
    return {
        "daily_calorie_target_kcal": pred,
        "bmr_kcal":  data["bmr_kcal"],
        "tdee_kcal": data["tdee_kcal"],
        "macros": {
            "protein_g":       round((cal * p) / 4),
            "carbohydrates_g": round((cal * c) / 4),
            "fat_g":           round((cal * f) / 9),
        },
    }

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run("app_firebase:app", host="0.0.0.0", port=8000, reload=True)
