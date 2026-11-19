"""Entrenamiento y evaluación de un Isolation Forest con tracking en MLflow.

Ejecuta varias configuraciones (runs) dentro de un mismo experimento,
registra parámetros, métricas, artefactos y el modelo, y finalmente
registra el mejor run en el Model Registry con el alias "champion".
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME = "insurance-claims-anomaly-detection"
MODEL_NAME = "claims-anomaly-detector"
DATA_PATH = "data/synthetic_health_claims.csv"

NUMERIC = ["claim_amount", "num_services", "patient_age", "days_since_last_claim"]
CATEGORICAL = ["provider_id"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "is_anomaly"

# Hiperparámetros a comparar
CONFIGS = [
    {"n_estimators": 100, "contamination": 0.10, "max_samples": "auto"},
    {"n_estimators": 200, "contamination": 0.20, "max_samples": "auto"},
    {"n_estimators": 300, "contamination": 0.20, "max_samples": 64},
    {"n_estimators": 200, "contamination": 0.30, "max_samples": "auto"},
]


def build_pipeline(params: dict) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = IsolationForest(random_state=42, **params)
    return Pipeline([("preprocessor", pre), ("model", model)])


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[dict, list]:
    # IsolationForest: -1 = anomalía, 1 = normal  ->  1 = anomalía, 0 = normal
    y_pred = (pipe.predict(X) == -1).astype(int)
    # decision_function: valores más bajos = más anómalo -> invertimos para AUC
    scores = -pipe.decision_function(X)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "flagged_ratio": float(y_pred.mean()),
    }
    return metrics, y_pred


def plot_confusion(y_true, y_pred, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anómala"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusión (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    results = []
    for i, params in enumerate(CONFIGS, start=1):
        run_name = f"iforest-{i}-n{params['n_estimators']}-c{params['contamination']}"
        with mlflow.start_run(run_name=run_name) as run:
            pipe = build_pipeline(params)
            # Entrenamiento no supervisado: la etiqueta no se usa para ajustar
            pipe.fit(X_train)

            train_metrics, _ = evaluate(pipe, X_train, y_train)
            test_metrics, y_pred = evaluate(pipe, X_test, y_test)

            mlflow.log_params(params)
            mlflow.log_params(
                {"model_type": "IsolationForest", "test_size": 0.3,
                 "n_train": len(X_train), "n_test": len(X_test)}
            )
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.set_tags({"dataset": DATA_PATH, "task": "anomaly-detection"})

            cm_path = f"confusion_matrix_{i}.png"
            plot_confusion(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path, artifact_path="plots")
            os.remove(cm_path)

            signature = infer_signature(X_train, pipe.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=pipe,
                name="model",
                signature=signature,
                input_example=X_train.head(3),
                # Modelo entrenado por nosotros: confiamos en los árboles internos
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(f"{run_name}: " + ", ".join(f"{k}={v:.3f}" for k, v in test_metrics.items()))
            results.append((test_metrics["f1"], test_metrics["roc_auc"], run.info.run_id, run_name))

    # Seleccionar el mejor run por F1 (desempate por ROC-AUC)
    best_f1, best_auc, best_run_id, best_name = max(results)
    print(f"\nMejor run: {best_name} (F1={best_f1:.3f}, AUC={best_auc:.3f})")

    mv = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    client.update_model_version(
        MODEL_NAME, mv.version,
        description=f"Isolation Forest seleccionado del run {best_name} (F1 test={best_f1:.3f})",
    )
    print(f"Modelo registrado: {MODEL_NAME} v{mv.version} con alias 'champion'")


if __name__ == "__main__":
    main()
