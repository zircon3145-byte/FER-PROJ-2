#src/utils/mlflow_config.py
import mlflow

def setup_mlflow(experiment_name="FER-Emotion-Recognition"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment = mlflow.get_experiment_by_name(experiment_name)
