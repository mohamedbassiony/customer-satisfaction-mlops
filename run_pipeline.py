from pipelines.training_pipeline import train_pipeline
from zenml.client import Client


if __name__ == "__main__":

    # Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline()

#mlflow ui --backend-store-uri "file:/home/bassiony/.config/zenml/local_stores/66dc076c-2554-42c2-bbde-2e5129405d22/mlruns"