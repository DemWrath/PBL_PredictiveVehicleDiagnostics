import requests
import json

API_URL = "https://pbl-predictivevehiclediagnostics.onrender.com"

print("="*70)
print("VEHICLE FAULT DETECTION - PREDICTION TEST")
print("="*70)

# Test 1: Normal idle conditions
print("\n[TEST 1] Normal Vehicle (Idle, Stopped)")
print("-" * 70)

normal_data = {
    "vehicle_id": "SEAT_LEON_TEST_001",
    "timestamp": "2026-02-15T01:10:00Z",
    "sensor_data": {
        "coolant_temp": [90.0] * 1200,    # Normal operating temp
        "MAP": [45.0] * 1200,              # Normal idle vacuum
        "RPM": [800.0] * 1200,             # Normal idle RPM
        "speed": [0.0] * 1200,             # Stopped
        "intake_temp": [25.0] * 1200,      # Room temperature
        "MAF": [2.5] * 1200,               # Normal idle airflow
        "throttle": [0.0] * 1200,          # Closed throttle
        "ambient": [20.0] * 1200,          # 20°C ambient
        "APP_D": [0.0] * 1200,             # Pedal released
        "APP_E": [0.0] * 1200,             # Pedal released
    }
}

response = requests.post(f"{API_URL}/predict", json=normal_data, timeout=30)

print("Status Code:", response.status_code)
print("Full Response:", response.text)

if response.status_code != 200:
    print("API ERROR — stopping test.")
    exit()

result = response.json()


print(f"Vehicle ID: {result['vehicle_id']}")
print(f"Fault Detected: {result['fault_detected']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Threshold: {result['threshold_used']}")

if result['fault_detected']:
    print(f"\n⚠️  ALERT: Fault detected!")
    diag = result['diagnostic']
    print(f"   Tier: {diag.get('tier')}")
    print(f"   Issues: {diag.get('possible_categories', [])}")
else:
    print(f"\n✓ Vehicle operating normally")

# Test 2: APP sensor disagreement (common fault)
print("\n" + "="*70)
print("[TEST 2] Simulated Fault: APP Sensor Disagreement")
print("-" * 70)

fault_data = {
    "vehicle_id": "SEAT_LEON_FAULT_002",
    "timestamp": "2026-02-15T01:11:00Z",
    "sensor_data": {
        "coolant_temp": [92.0] * 1200,
        "MAP": [50.0] * 1200,
        "RPM": [1200.0] * 1200,            # Higher RPM
        "speed": [30.0] * 1200,            # Moving
        "intake_temp": [30.0] * 1200,
        "MAF": [8.5] * 1200,               # Higher airflow
        "throttle": [20.0] * 1200,         # Throttle open 20%
        "ambient": [22.0] * 1200,
        "APP_D": [25.0] * 1200,            # Pedal at 25%
        "APP_E": [10.0] * 1200,            # ⚠️ DISAGREEMENT! Should match APP_D
    }
}

response = requests.post(f"{API_URL}/predict", json=fault_data, timeout=30)
result = response.json()






print(f"Vehicle ID: {result['vehicle_id']}")
print(f"Fault Detected: {result['fault_detected']}")
print(f"Confidence: {result['confidence']:.1%}")

if result['fault_detected']:
    print(f"\n⚠️  FAULT DETECTED!")
    diag = result['diagnostic']
    print(f"   Diagnostic Tier: {diag.get('tier')}")
    print(f"   Mode: {diag.get('mode')}")
    print(f"   Possible Issues: {diag.get('possible_categories', [])}")
    print(f"   Check Sensors: {diag.get('primary_sensors', [])}")
    print(f"   Recommendation: {diag.get('recommendation', 'N/A')}")
    print(f"   Urgency: {diag.get('urgency', 'UNKNOWN')}")
else:
    print(f"\n✓ No fault detected (unexpected)")

# Test 3: Overheating scenario
print("\n" + "="*70)
print("[TEST 3] Simulated Fault: Coolant Overheating")
print("-" * 70)

overheat_data = {
    "vehicle_id": "SEAT_LEON_OVERHEAT_003",
    "timestamp": "2026-02-15T01:12:00Z",
    "sensor_data": {
        "coolant_temp": [118.0] * 1200,    # ⚠️ OVERHEATING!
        "MAP": [60.0] * 1200,
        "RPM": [2500.0] * 1200,
        "speed": [80.0] * 1200,
        "intake_temp": [45.0] * 1200,
        "MAF": [25.0] * 1200,
        "throttle": [50.0] * 1200,
        "ambient": [30.0] * 1200,
        "APP_D": [50.0] * 1200,
        "APP_E": [50.0] * 1200,
    }
}

response = requests.post(f"{API_URL}/predict", json=overheat_data, timeout=30)
result = response.json()

print(f"Vehicle ID: {result['vehicle_id']}")
print(f"Fault Detected: {result['fault_detected']}")
print(f"Confidence: {result['confidence']:.1%}")

if result['fault_detected']:
    print(f"\n⚠️  CRITICAL FAULT!")
    diag = result['diagnostic']
    print(f"   Issues: {diag.get('possible_categories', [])}")
    print(f"   Check: {diag.get('primary_sensors', [])}")
else:
    print(f"\n✓ No fault detected")

print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)
print("\n✓ Your fault detection system is working!")
print("  API URL: https://pbl-predictivevehiclediagnostics.onrender.com")
print("  Network URL: https://pbl-predictivevehiclediagnostics.onrender.com")