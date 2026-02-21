"""
=============================================================
  API Test Client — Health & Nutrition Advisor
=============================================================
Tests all endpoints with a sample user profile.

Run AFTER starting the API:
  uvicorn app:app --reload --port 8000

Then in a new terminal:
  python test_api.py
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"

# ── Shared sample user payload ────────────────────────────────────────────────
SAMPLE_USER = {
    "age":                    28,
    "gender":                 "Female",
    "height_cm":              162.0,
    "weight_kg":              68.0,
    "activity_level":         "Moderately Active",
    "dietary_preference":     "Vegetarian",
    "health_goal":            "Weight Loss",
    "food_allergy":           "None",
    "medical_history":        "None",
    "avg_sleep_hours":        6.5,
    "daily_steps":            7200,
    "water_intake_litres":    2.2,
    "calories_burned_per_day": 1950,
    "avg_heart_rate_bpm":     76,
    "stress_level":           "Medium",
    "sleep_quality_score":    "Fair",
    "fitness_level":          "Beginner",
    "workout_days_per_week":  3,
    "meal_frequency_per_day": 4,
    "current_streak_days":    12,
    "total_workouts_logged":  35,
    "total_meals_logged":     88,
    "number_of_people":       2,
    "budget_range":           "medium",
}

ENDPOINTS = [
    ("GET",  "/",                    None),
    ("GET",  "/models",              None),
    ("POST", "/predict/obesity-risk",      SAMPLE_USER),
    ("POST", "/predict/diabetes-risk",     SAMPLE_USER),
    ("POST", "/predict/hypertension-risk", SAMPLE_USER),
    ("POST", "/predict/bmi-category",      SAMPLE_USER),
    ("POST", "/predict/badge-status",      SAMPLE_USER),
    ("POST", "/predict/calorie-target",    SAMPLE_USER),
    ("POST", "/predict/all",               SAMPLE_USER),
]

def call(method, path, body=None):
    url  = BASE_URL + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req  = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as ex:
        return 0, {"error": str(ex)}

def pretty(d, indent=2):
    return json.dumps(d, indent=indent, ensure_ascii=False)

# ── Run tests ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Health & Nutrition Advisor — API Test Suite")
print("=" * 65)

all_passed = True

for method, path, body in ENDPOINTS:
    print(f"\n{'─'*65}")
    print(f"  {method} {path}")
    print(f"{'─'*65}")

    status, response = call(method, path, body)

    if status == 200:
        print(f"  ✅ Status: {status}")
        # Print key fields only for large responses
        if path == "/predict/all":
            print(f"  User Stats     : {response.get('user_stats')}")
            print(f"  Predictions    : {pretty(response.get('predictions'), indent=4)}")
        elif path == "/models":
            for task, info in response.items():
                print(f"    {task:30s} → {info['algorithm']:25s} score={info['score']}")
        elif "/predict/" in path:
            print(f"  Prediction     : {response.get('prediction')}")
            if "probabilities" in response and response["probabilities"]:
                print(f"  Probabilities  : {response['probabilities']}")
            if "macros" in response:
                print(f"  Macros         : {response['macros']}")
            if "bmi_value" in response:
                print(f"  BMI Value      : {response['bmi_value']}")
        else:
            print(f"  Response       : {pretty(response)}")
    else:
        print(f"  ❌ Status: {status}")
        print(f"  Error   : {pretty(response)}")
        all_passed = False

print(f"\n{'='*65}")
if all_passed:
    print("  ✅  All tests passed! API is working correctly.")
else:
    print("  ❌  Some tests failed. Check the API is running on port 8000.")
print(f"{'='*65}\n")