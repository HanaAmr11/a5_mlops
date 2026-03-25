import os
import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI is not set.")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("A5_MLOps_Pipeline")

with mlflow.start_run() as run:
    # accuracy = 0.92
    accuracy = 0.80

    mlflow.log_metric("accuracy", accuracy)

    run_id = run.info.run_id
    with open("model_info.txt", "w", encoding="utf-8") as f:
        f.write(run_id)

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")
    print("model_info.txt created successfully")