"""Servicio BentoML para revisar reclamaciones de seguros.

Endpoints:
  POST /predict      -> lista de reclamaciones en JSON
  POST /upload_csv   -> archivo CSV con reclamaciones (carga masiva)
  POST /health_check -> estado del servicio
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import bentoml
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel, Field

MODEL_NAME = "claims-anomaly-detector"
FEATURES = ["claim_amount", "num_services", "patient_age",
            "provider_id", "days_since_last_claim"]


class Claim(BaseModel):
    claim_id: str = Field(examples=["CLM-9001"])
    claim_amount: float = Field(gt=0, examples=[1250.50])
    num_services: int = Field(ge=1, examples=[2])
    patient_age: int = Field(ge=0, le=120, examples=[45])
    provider_id: Literal["P001", "P002", "P003", "P004", "P005"] = "P001"
    days_since_last_claim: int = Field(ge=0, examples=[120])


class ClaimResult(BaseModel):
    claim_id: str
    anomaly_score: float
    is_anomaly: bool
    decision: str


class BatchResult(BaseModel):
    total_claims: int
    flagged_for_review: int
    auto_approved: int
    results: List[ClaimResult]


@bentoml.service(
    name="claims_review_service",
    traffic={"timeout": 30},
    resources={"cpu": "1"},
)
class ClaimsReviewService:
    bento_model = bentoml.models.BentoModel(f"{MODEL_NAME}:latest")

    def __init__(self) -> None:
        # Carga el pipeline sklearn nativo para acceder también a decision_function
        self.model = mlflow.sklearn.load_model(self.bento_model.path_of("mlflow_model"))

    def _score(self, df: pd.DataFrame) -> BatchResult:
        missing = [c for c in ["claim_id", *FEATURES] if c not in df.columns]
        if missing:
            raise bentoml.exceptions.InvalidArgument(f"Faltan columnas: {missing}")
        X = df[FEATURES]
        preds = self.model.predict(X)              # -1 anomalía, 1 normal
        scores = -self.model.decision_function(X)  # mayor = más anómalo
        results = [
            ClaimResult(
                claim_id=str(cid),
                anomaly_score=round(float(s), 4),
                is_anomaly=bool(p == -1),
                decision="REVISIÓN MANUAL" if p == -1 else "APROBACIÓN AUTOMÁTICA",
            )
            for cid, p, s in zip(df["claim_id"], preds, scores)
        ]
        flagged = sum(r.is_anomaly for r in results)
        return BatchResult(
            total_claims=len(results),
            flagged_for_review=flagged,
            auto_approved=len(results) - flagged,
            results=results,
        )

    @bentoml.api
    def predict(self, claims: List[Claim]) -> BatchResult:
        df = pd.DataFrame([c.model_dump() for c in claims])
        return self._score(df)

    @bentoml.api
    def upload_csv(self, file: Path) -> BatchResult:
        df = pd.read_csv(file)
        return self._score(df)

    @bentoml.api
    def health_check(self) -> dict:
        return {"status": "ok", "model": str(self.bento_model.tag)}
