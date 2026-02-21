"""
=============================================================
  AI Health & Nutrition Advisor — ML Training Pipeline
=============================================================
Targets trained:
  1. obesity_risk         → Multiclass Classification (Low/Moderate/High)
  2. diabetes_risk        → Multiclass Classification
  3. hypertension_risk    → Multiclass Classification
  4. bmi_category         → Multiclass Classification
  5. daily_calorie_target → Regression
  6. badge_status         → Multiclass Classification

Run:
    python train_health_model.py

Outputs (saved in same folder):
  - models/          → All trained .pkl model files
  - reports/         → Classification reports & metrics
  - plots/           → Feature importance & confusion matrix plots
"""

import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score, mean_squared_error)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "health_nutrition_dataset_2000.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")

for d in [MODEL_DIR, REPORT_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 1 — Loading Dataset")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"  Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Null values   : {df.isnull().sum().sum()}")
print(f"  Duplicates    : {df.duplicated().sum()}")
print(f"\n  Column dtypes :\n{df.dtypes.value_counts().to_string()}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 2 — Preprocessing")
print("=" * 65)

# Drop columns that are identifiers / leakage / free-text
DROP_COLS = [
    "user_id", "full_name", "email", "phone", "signup_date",
    "city", "state", "bedtime", "wake_time",   # high-cardinality / non-numeric
    "available_ingredients",                    # free text
    "weight_7day_log_kg", "weight_30day_log_kg" # data leakage for risk targets
]

df_clean = df.drop(columns=DROP_COLS, errors="ignore")

# Parse budget_range → numeric midpoint
budget_map = {
    "Low (< ₹2000/week)": 1500,
    "Medium (₹2000–5000/week)": 3500,
    "High (₹5000–10000/week)": 7500,
    "Premium (> ₹10000/week)": 12000,
}
df_clean["budget_numeric"] = df_clean["budget_range"].map(budget_map)
df_clean.drop(columns=["budget_range"], inplace=True)

# Identify column types
NUMERIC_COLS = df_clean.select_dtypes(include=[np.number]).columns.tolist()
CATEG_COLS   = df_clean.select_dtypes(include=["object"]).columns.tolist()

print(f"  Numeric columns : {len(NUMERIC_COLS)}")
print(f"  Category columns: {len(CATEG_COLS)}")
print(f"  Category cols   : {CATEG_COLS}\n")

# Label-encode all categoricals for use as features
le_dict = {}
df_encoded = df_clean.copy()
for col in CATEG_COLS:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_clean[col].astype(str))
    le_dict[col] = le

# Save label encoders
with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump(le_dict, f)
print("  ✓ Label encoders saved")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DEFINE TASKS
# ══════════════════════════════════════════════════════════════════════════════

# Features used across classification tasks
CLASSIFICATION_FEATURES = [
    "age", "gender", "height_cm", "weight_kg", "bmi",
    "activity_level", "dietary_preference", "health_goal",
    "food_allergy", "medical_history", "avg_sleep_hours",
    "daily_steps", "water_intake_litres", "calories_burned_per_day",
    "avg_heart_rate_bpm", "stress_level", "fitness_level",
    "workout_days_per_week", "current_streak_days", "meal_frequency_per_day",
    "daily_calorie_target", "protein_target_g", "carbs_target_g", "fat_target_g",
    "total_workouts_logged", "total_meals_logged", "number_of_people",
    "sleep_quality_score", "budget_numeric"
]

REGRESSION_FEATURES = [
    "age", "gender", "height_cm", "weight_kg", "bmi",
    "activity_level", "health_goal", "dietary_preference",
    "bmr_kcal", "tdee_kcal", "avg_sleep_hours", "daily_steps",
    "calories_burned_per_day", "stress_level", "fitness_level",
    "workout_days_per_week", "meal_frequency_per_day",
    "water_intake_litres", "avg_heart_rate_bpm"
]

TASKS = {
    # name           : (target_col, type,           features)
    "obesity_risk"      : ("obesity_risk",      "classification", CLASSIFICATION_FEATURES),
    "diabetes_risk"     : ("diabetes_risk",     "classification", CLASSIFICATION_FEATURES),
    "hypertension_risk" : ("hypertension_risk", "classification", CLASSIFICATION_FEATURES),
    "bmi_category"      : ("bmi_category",      "classification", REGRESSION_FEATURES),
    "badge_status"      : ("badge_status",      "classification", CLASSIFICATION_FEATURES),
    "calorie_target"    : ("daily_calorie_target", "regression",  REGRESSION_FEATURES),
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. HELPER — PLOT CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, classes, task_name, model_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix\n{task_name} — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"cm_{task_name}_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()

def plot_feature_importance(importances, feature_names, task_name, model_name, top_n=15):
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), importances[indices][::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances\n{task_name} — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"fi_{task_name}_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()

def plot_regression_actual_vs_pred(y_test, y_pred, task_name, model_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.35, color="#0D9488", edgecolors="white", linewidths=0.3, s=30)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(f"Actual vs Predicted\n{task_name} — {model_name}", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"reg_{task_name}_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN ALL TASKS
# ══════════════════════════════════════════════════════════════════════════════

all_results = {}

for task_name, (target_col, task_type, feature_list) in TASKS.items():
    print("\n" + "=" * 65)
    print(f"  TASK : {task_name.upper()}")
    print(f"  Type : {task_type}  |  Target: {target_col}")
    print("=" * 65)

    # ── Prepare X / y ────────────────────────────────────────────────────────
    available_feats = [f for f in feature_list if f in df_encoded.columns and f != target_col]
    X = df_encoded[available_feats].values
    y_raw = df_encoded[target_col].values if target_col in df_encoded.columns else df_clean[target_col].values

    # For regression target use original numeric values (not encoded)
    if task_type == "regression":
        y = df_clean[target_col].astype(float).values
    else:
        # Encode target if not already numeric
        if df_clean[target_col].dtype == object:
            le_target = LabelEncoder()
            y = le_target.fit_transform(df_clean[target_col].astype(str))
            target_classes = le_target.classes_
        else:
            y = df_encoded[target_col].values
            target_classes = le_dict.get(target_col, LabelEncoder()).classes_ if target_col in le_dict else np.unique(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

    # ── Define models ─────────────────────────────────────────────────────────
    if task_type == "classification":
        models = {
            "Random Forest"         : RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
            "Gradient Boosting"     : GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
            "Logistic Regression"   : LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "Decision Tree"         : DecisionTreeClassifier(max_depth=10, random_state=42),
            "K-Nearest Neighbors"   : KNeighborsClassifier(n_neighbors=7),
            "SVM"                   : SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        }
    else:
        models = {
            "Random Forest Regressor" : RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
            "Gradient Boosting"       : GradientBoostingClassifier.__new__(GradientBoostingClassifier),  # placeholder
            "Ridge Regression"        : Ridge(alpha=1.0),
        }
        # Fix: replace GBR placeholder with actual
        from sklearn.ensemble import GradientBoostingRegressor
        models["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)

    # ── Train & Evaluate ──────────────────────────────────────────────────────
    task_results = {}
    best_score   = -np.inf
    best_model   = None
    best_name    = None

    for model_name, model in models.items():
        print(f"\n  ▶ Training: {model_name}")

        # Use scaled data for LR, KNN, SVM; raw for tree-based
        use_scaled = model_name in ["Logistic Regression", "K-Nearest Neighbors", "SVM", "Ridge Regression"]
        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc  if use_scaled else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        if task_type == "classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="accuracy")
            test_acc  = (y_pred == y_test).mean()
            report    = classification_report(y_test, y_pred, target_names=[str(c) for c in target_classes], output_dict=True)

            print(f"    CV Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"    Test Accuracy : {test_acc:.4f}")
            print(f"    Macro F1      : {report['macro avg']['f1-score']:.4f}")

            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm, [str(c) for c in target_classes], task_name, model_name)

            score = test_acc
            task_results[model_name] = {
                "cv_accuracy_mean" : round(cv_scores.mean(), 4),
                "cv_accuracy_std"  : round(cv_scores.std(), 4),
                "test_accuracy"    : round(test_acc, 4),
                "macro_f1"         : round(report["macro avg"]["f1-score"], 4),
                "macro_precision"  : round(report["macro avg"]["precision"], 4),
                "macro_recall"     : round(report["macro avg"]["recall"], 4),
            }

        else:
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2   = r2_score(y_test, y_pred)

            print(f"    MAE  : {mae:.2f}")
            print(f"    RMSE : {rmse:.2f}")
            print(f"    R²   : {r2:.4f}")

            plot_regression_actual_vs_pred(y_test, y_pred, task_name, model_name)

            score = r2
            task_results[model_name] = {
                "mae"  : round(mae, 2),
                "rmse" : round(rmse, 2),
                "r2"   : round(r2, 4),
            }

        # Feature importance plot (tree-based models)
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(model.feature_importances_, available_feats, task_name, model_name)

        # Track best
        if score > best_score:
            best_score = score
            best_model = model
            best_name  = model_name

    # ── Save best model ───────────────────────────────────────────────────────
    model_path  = os.path.join(MODEL_DIR, f"{task_name}_best_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{task_name}_scaler.pkl")
    meta_path   = os.path.join(MODEL_DIR, f"{task_name}_meta.json")

    with open(model_path, "wb") as f:  pickle.dump(best_model, f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

    meta = {
        "task"       : task_name,
        "task_type"  : task_type,
        "target"     : target_col,
        "best_model" : best_name,
        "best_score" : round(best_score, 4),
        "features"   : available_feats,
    }
    with open(meta_path, "w") as f: json.dump(meta, f, indent=2)

    print(f"\n  ★ Best model: {best_name}  (score={best_score:.4f})")
    print(f"    Saved → {model_path}")

    all_results[task_name] = {"best_model": best_name, "score": round(best_score, 4), "models": task_results}

# ══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 6 — Final Summary")
print("=" * 65)

report_rows = []
for task, res in all_results.items():
    for model_name, metrics in res["models"].items():
        row = {"task": task, "model": model_name}
        row.update(metrics)
        report_rows.append(row)

summary_df = pd.DataFrame(report_rows)
report_path = os.path.join(REPORT_DIR, "model_comparison_report.csv")
summary_df.to_csv(report_path, index=False)
print(f"\n  Full report saved → {report_path}")

print("\n  ┌──────────────────────────────────────────────────────────────┐")
print("  │                 BEST MODEL PER TASK                         │")
print("  ├─────────────────────────┬──────────────────────┬────────────┤")
print("  │ Task                    │ Best Model           │ Score      │")
print("  ├─────────────────────────┼──────────────────────┼────────────┤")
for task, res in all_results.items():
    t = task[:24].ljust(24)
    m = res["best_model"][:21].ljust(21)
    s = str(res["score"]).ljust(10)
    print(f"  │ {t} │ {m} │ {s} │")
print("  └─────────────────────────┴──────────────────────┴────────────┘")

# ══════════════════════════════════════════════════════════════════════════════
# 7. EXAMPLE PREDICTION (using saved best models)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 7 — Example Prediction (Obesity Risk)")
print("=" * 65)

with open(os.path.join(MODEL_DIR, "obesity_risk_best_model.pkl"), "rb") as f:
    saved_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "obesity_risk_scaler.pkl"), "rb") as f:
    saved_scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "obesity_risk_meta.json")) as f:
    saved_meta = json.load(f)

# Build a sample input aligned to the feature list
sample_input = {
    "age": 35, "gender": 1, "height_cm": 170, "weight_kg": 85,
    "bmi": 29.4, "activity_level": 0, "dietary_preference": 3,
    "health_goal": 0, "food_allergy": 0, "medical_history": 0,
    "avg_sleep_hours": 5.5, "daily_steps": 3500,
    "water_intake_litres": 1.5, "calories_burned_per_day": 1800,
    "avg_heart_rate_bpm": 88, "stress_level": 2, "fitness_level": 0,
    "workout_days_per_week": 1, "current_streak_days": 2,
    "meal_frequency_per_day": 3, "daily_calorie_target": 2800,
    "protein_target_g": 140, "carbs_target_g": 245, "fat_target_g": 78,
    "total_workouts_logged": 12, "total_meals_logged": 45,
    "number_of_people": 3, "sleep_quality_score": 0, "budget_numeric": 3500
}

feats = saved_meta["features"]
X_sample = np.array([[sample_input.get(f, 0) for f in feats]])
pred = saved_model.predict(X_sample)[0]
if hasattr(saved_model, "predict_proba"):
    proba = saved_model.predict_proba(X_sample)[0]
    classes = le_dict["obesity_risk"].classes_ if "obesity_risk" in le_dict else ["High","Low","Moderate"]
    print(f"\n  Sample input → BMI={sample_input['bmi']}, Steps={sample_input['daily_steps']}, Sleep={sample_input['avg_sleep_hours']}h")
    print(f"  Predicted obesity risk class (encoded) : {pred}")
    print(f"  Class probabilities:")
    for cls, p in zip(classes, proba):
        print(f"    {cls:12s}: {p:.3f}")

print("\n" + "=" * 65)
print("  ✅ Training complete! All outputs saved to:")
print(f"     models/  → {MODEL_DIR}")
print(f"     plots/   → {PLOT_DIR}")
print(f"     reports/ → {REPORT_DIR}")
print("=" * 65)
