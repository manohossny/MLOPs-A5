import sys

THRESHOLD = 0.85

with open("model_info.txt") as f:
    run_id = f.read().strip()

with open("accuracy.txt") as f:
    accuracy = float(f.read().strip())

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Threshold: {THRESHOLD}")

if accuracy >= THRESHOLD:
    print(f"PASS: {accuracy:.4f} >= {THRESHOLD}")
    sys.exit(0)
else:
    print(f"FAIL: {accuracy:.4f} < {THRESHOLD}")
    sys.exit(1)
