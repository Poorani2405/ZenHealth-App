"""
=============================================================
  ML Model Validation — JSON Test Cases
=============================================================
Tests 6 different user profiles, each expecting a specific
prediction outcome. Run this to verify models are correct.

Run:
  python test_model_validation.py
=============================================================
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"

# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES — Each has an expected prediction outcome
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [

    # ── TEST 1: Obese, Sedentary, Poor Health → HIGH RISK across all ──────────
    {
        "name": "TEST 1 — High Risk Patient (Obese + Sedentary + Diabetic)",
        "expected": {
            "obesity_risk":     "High",
            "diabetes_risk":    "High",
            "hypertension_risk":"High",
            "bmi_category":     "Obese",
            "badge_status":     "Beginner",
        },
        "payload": {
            "age":                    52,
            "gender":                 "Male",
            "height_cm":              168.0,
            "weight_kg":              102.0,
            "activity_level":         "Sedentary",
            "dietary_preference":     "Non-Vegetarian",
            "health_goal":            "Weight Loss",
            "food_allergy":           "None",
            "medical_history":        "Diabetes Type 2",
            "avg_sleep_hours":        4.5,
            "daily_steps":            1500,
            "water_intake_litres":    1.0,
            "calories_burned_per_day":1400,
            "avg_heart_rate_bpm":     98,
            "stress_level":           "High",
            "sleep_quality_score":    "Poor",
            "fitness_level":          "Beginner",
            "workout_days_per_week":  0,
            "meal_frequency_per_day": 2,
            "current_streak_days":    0,
            "total_workouts_logged":  2,
            "total_meals_logged":     8,
            "number_of_people":       1,
            "budget_range":           "low",
        }
    },

    # ── TEST 2: Fit, Active, Healthy → LOW RISK across all ────────────────────
    {
        "name": "TEST 2 — Low Risk (Fit + Active + Healthy)",
        "expected": {
            "obesity_risk":     "Low",
            "diabetes_risk":    "Low",
            "hypertension_risk":"Low",
            "bmi_category":     "Normal",
            "badge_status":     "Gold",
        },
        "payload": {
            "age":                    26,
            "gender":                 "Female",
            "height_cm":              165.0,
            "weight_kg":              57.0,
            "activity_level":         "Very Active",
            "dietary_preference":     "Mediterranean",
            "health_goal":            "Maintain Weight",
            "food_allergy":           "None",
            "medical_history":        "None",
            "avg_sleep_hours":        8.0,
            "daily_steps":            12000,
            "water_intake_litres":    3.5,
            "calories_burned_per_day":2800,
            "avg_heart_rate_bpm":     62,
            "stress_level":           "Low",
            "sleep_quality_score":    "Excellent",
            "fitness_level":          "Advanced",
            "workout_days_per_week":  6,
            "meal_frequency_per_day": 5,
            "current_streak_days":    65,
            "total_workouts_logged":  200,
            "total_meals_logged":     400,
            "number_of_people":       2,
            "budget_range":           "high",
        }
    },

    # ── TEST 3: Underweight, Young ─────────────────────────────────────────────
    {
        "name": "TEST 3 — Underweight Young Adult",
        "expected": {
            "bmi_category":     "Underweight",
            "obesity_risk":     "Low",
            "badge_status":     "Beginner",
        },
        "payload": {
            "age":                    19,
            "gender":                 "Female",
            "height_cm":              160.0,
            "weight_kg":              42.0,
            "activity_level":         "Lightly Active",
            "dietary_preference":     "Vegan",
            "health_goal":            "Muscle Gain",
            "food_allergy":           "Dairy",
            "medical_history":        "Anemia",
            "avg_sleep_hours":        7.0,
            "daily_steps":            5000,
            "water_intake_litres":    2.0,
            "calories_burned_per_day":1600,
            "avg_heart_rate_bpm":     70,
            "stress_level":           "Low",
            "sleep_quality_score":    "Good",
            "fitness_level":          "Beginner",
            "workout_days_per_week":  2,
            "meal_frequency_per_day": 3,
            "current_streak_days":    3,
            "total_workouts_logged":  10,
            "total_meals_logged":     20,
            "number_of_people":       1,
            "budget_range":           "low",
        }
    },

    # ── TEST 4: Overweight, Moderate Risk ─────────────────────────────────────
    {
        "name": "TEST 4 — Overweight, Moderate Risk (Middle-aged)",
        "expected": {
            "bmi_category":     "Overweight",
            "obesity_risk":     "Moderate",
            "badge_status":     "Bronze",
        },
        "payload": {
            "age":                    40,
            "gender":                 "Male",
            "height_cm":              175.0,
            "weight_kg":              88.0,
            "activity_level":         "Lightly Active",
            "dietary_preference":     "Non-Vegetarian",
            "health_goal":            "Weight Loss",
            "food_allergy":           "None",
            "medical_history":        "High Cholesterol",
            "avg_sleep_hours":        6.0,
            "daily_steps":            4500,
            "water_intake_litres":    1.8,
            "calories_burned_per_day":1900,
            "avg_heart_rate_bpm":     82,
            "stress_level":           "Medium",
            "sleep_quality_score":    "Fair",
            "fitness_level":          "Beginner",
            "workout_days_per_week":  2,
            "meal_frequency_per_day": 3,
            "current_streak_days":    8,
            "total_workouts_logged":  25,
            "total_meals_logged":     60,
            "number_of_people":       4,
            "budget_range":           "medium",
        }
    },

    # ── TEST 5: Platinum Badge (Long streak, very active) ─────────────────────
    {
        "name": "TEST 5 — Platinum Badge (High Streak + Max Activity)",
        "expected": {
            "badge_status":     "Platinum",
            "obesity_risk":     "Low",
            "bmi_category":     "Normal",
        },
        "payload": {
            "age":                    33,
            "gender":                 "Male",
            "height_cm":              178.0,
            "weight_kg":              74.0,
            "activity_level":         "Extremely Active",
            "dietary_preference":     "Keto",
            "health_goal":            "Muscle Gain",
            "food_allergy":           "None",
            "medical_history":        "None",
            "avg_sleep_hours":        7.5,
            "daily_steps":            15000,
            "water_intake_litres":    4.0,
            "calories_burned_per_day":3500,
            "avg_heart_rate_bpm":     58,
            "stress_level":           "Low",
            "sleep_quality_score":    "Excellent",
            "fitness_level":          "Advanced",
            "workout_days_per_week":  7,
            "meal_frequency_per_day": 5,
            "current_streak_days":    110,
            "total_workouts_logged":  290,
            "total_meals_logged":     550,
            "number_of_people":       1,
            "budget_range":           "premium",
        }
    },

    # ── TEST 6: High Calorie Need (Athlete) ────────────────────────────────────
    {
        "name": "TEST 6 — High Calorie Target (Athlete, Muscle Gain)",
        "expected": {
            "calorie_target_min": 3000,
            "bmi_category":       "Normal",
            "badge_status":       "Silver",
        },
        "payload": {
            "age":                    24,
            "gender":                 "Male",
            "height_cm":              185.0,
            "weight_kg":              82.0,
            "activity_level":         "Extremely Active",
            "dietary_preference":     "Non-Vegetarian",
            "health_goal":            "Muscle Gain",
            "food_allergy":           "None",
            "medical_history":        "None",
            "avg_sleep_hours":        8.5,
            "daily_steps":            14000,
            "water_intake_litres":    4.5,
            "calories_burned_per_day":3800,
            "avg_heart_rate_bpm":     55,
            "stress_level":           "Low",
            "sleep_quality_score":    "Excellent",
            "fitness_level":          "Advanced",
            "workout_days_per_week":  6,
            "meal_frequency_per_day": 5,
            "current_streak_days":    35,
            "total_workouts_logged":  150,
            "total_meals_logged":     280,
            "number_of_people":       1,
            "budget_range":           "high",
        }
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def call_api(payload):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{BASE_URL}/predict/all",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as ex:
        return 0, {"error": str(ex)}

def check(actual, expected_key, expected_val, actual_val):
    if expected_key == "calorie_target_min":
        passed = float(actual_val) >= expected_val
    else:
        passed = str(actual_val).lower() == str(expected_val).lower()
    symbol = "✅" if passed else "❌"
    print(f"    {symbol} {expected_key:25s} Expected: {str(expected_val):12s} Got: {actual_val}")
    return passed

# ── Run all tests ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ML MODEL VALIDATION — JSON TEST SUITE")
print("=" * 65)

total_checks = 0
passed_checks = 0

for i, tc in enumerate(TEST_CASES):
    print(f"\n{'─'*65}")
    print(f"  {tc['name']}")
    print(f"{'─'*65}")

    status, response = call_api(tc["payload"])

    if status != 200:
        print(f"  ❌ API Error {status}: {response}")
        continue

    preds      = response.get("predictions", {})
    user_stats = response.get("user_stats", {})

    print(f"  BMI: {user_stats.get('bmi')}  |  "
          f"BMR: {user_stats.get('bmr_kcal')} kcal  |  "
          f"TDEE: {user_stats.get('tdee_kcal')} kcal")
    print(f"  Calorie Target: {preds.get('daily_calorie_target_kcal')} kcal")
    print(f"  Macros → Protein: {preds.get('macros',{}).get('protein_g')}g  "
          f"Carbs: {preds.get('macros',{}).get('carbohydrates_g')}g  "
          f"Fat: {preds.get('macros',{}).get('fat_g')}g")
    print()

    for key, expected_val in tc["expected"].items():
        if key == "calorie_target_min":
            actual_val = preds.get("daily_calorie_target_kcal", 0)
        else:
            actual_val = preds.get(key, "N/A")

        result = check(response, key, expected_val, actual_val)
        total_checks  += 1
        passed_checks += 1 if result else 0

# ── Final Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  RESULTS: {passed_checks}/{total_checks} checks passed")
if passed_checks == total_checks:
    print("  🎉 ALL CHECKS PASSED — Models are correctly trained!")
elif passed_checks >= total_checks * 0.8:
    print("  ✅ MOSTLY PASSED — Models are working well (minor variance expected)")
else:
    print("  ⚠️  SOME CHECKS FAILED — Review model training data or features")
print(f"{'='*65}\n")
