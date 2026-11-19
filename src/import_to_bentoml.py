"""Importa el modelo 'champion' del MLflow Model Registry al Model Store de BentoML."""
import os

import bentoml
import mlflow

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MODEL_NAME = "claims-anomaly-detector"

mlflow.set_tracking_uri(TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}@champion"

bento_model = bentoml.mlflow.import_model(
    MODEL_NAME,
    model_uri=model_uri,
    labels={"stage": "champion", "framework": "sklearn"},
    metadata={"source": model_uri},
)
print(f"Modelo importado a BentoML: {bento_model.tag}")
