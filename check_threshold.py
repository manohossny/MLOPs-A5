import sys
import mlflow

THRESHOLD = 0.85

client = mlflow.tracking.MlflowClient()

with open("model_info.txt") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

run = client.get_run(run_id)
accuracy = run.data.metrics["accuracy"]

print(f"Accuracy: {accuracy:.4f}")
print(f"Threshold: {THRESHOLD}")

if accuracy >= THRESHOLD:
    print(f"PASS: {accuracy:.4f} >= {THRESHOLD}")
    sys.exit(0)
else:
    print(f"FAIL: {accuracy:.4f} < {THRESHOLD}")
    sys.exit(1)
