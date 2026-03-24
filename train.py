import os
import random
import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("A5_MLOps_Pipeline")

with mlflow.start_run() as run:
    # accuracy = random.uniform(0.80, 0.95)
    # accuracy = 0.80
    accuracy = 0.90

    mlflow.log_metric("accuracy", accuracy)

    run_id = run.info.run_id

    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print("model_info.txt created successfully")