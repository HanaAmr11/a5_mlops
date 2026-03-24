import os
import sys
import mlflow

THRESHOLD = 0.85

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy is None:
    print("Error: accuracy metric not found.")
    sys.exit(1)

if accuracy < THRESHOLD:
    print("Model accuracy is below threshold. Deployment failed.")
    sys.exit(1)

print("Model passed threshold. Deployment can continue.")