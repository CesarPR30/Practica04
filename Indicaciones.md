Construcción de un pipeline de Machine Learning de extremo a extremo

En esta práctica se requiere utilizar todas las herramientas que hemos explorado hasta ahora para construir un pipeline completo de Machine Learning, desde el desarrollo hasta su puesta en producción. Este pipeline debe incluir un entorno para desarrollar el modelo, un lugar donde almacenar nuestros experimentos y, finalmente, un componente para servir el modelo y realizar predicciones.

CASO: Procesamiento tradicional de reclamaciones de seguros
Flujo de revisión y aprobación documental.png

Entregables
El objetivo es desarrollar un servicio que permita cargar reclamaciones, analizarlas mediante un modelo de Machine Learning e identificar los casos que requieren una revisión adicional.

Generación de datos sintéticos para el proyecto: (code)
import numpy as np
import pandas as pd

# Reproducibility
rng = np.random.default_rng(42)

NORMAL_CLAIMS = 120
ANOMALOUS_CLAIMS = 30
TOTAL = NORMAL_CLAIMS + ANOMALOUS_CLAIMS

# Generate normal claims
normal = pd.DataFrame({
    "claim_amount": rng.uniform(100, 3000, NORMAL_CLAIMS),
    "num_services": rng.integers(1, 6, NORMAL_CLAIMS),
    "patient_age": rng.integers(18, 85, NORMAL_CLAIMS),
    "provider_id": rng.choice(
        ["P001", "P002", "P003", "P004", "P005"],
        NORMAL_CLAIMS
    ),
    "days_since_last_claim": rng.integers(
        30, 365, NORMAL_CLAIMS
    ),
    "is_anomaly": 0
})

# Generate anomalous claims
anomalous = pd.DataFrame({
    "claim_amount": rng.uniform(
        8000, 25000, ANOMALOUS_CLAIMS
    ),
    "num_services": rng.integers(
        8, 20, ANOMALOUS_CLAIMS
    ),
    "patient_age": rng.integers(
        18, 85, ANOMALOUS_CLAIMS
    ),
    "provider_id": rng.choice(
        ["P001", "P002", "P003", "P004", "P005"],
        ANOMALOUS_CLAIMS
    ),
    "days_since_last_claim": rng.integers(
        0, 15, ANOMALOUS_CLAIMS
    ),
    "is_anomaly": 1
})

# Combine and shuffle
df = pd.concat(
    [normal, anomalous],
    ignore_index=True
)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Create unique claim identifiers
df.insert(
    0,
    "claim_id",
    [f"CLM-{i:04d}" for i in range(1, TOTAL + 1)]
)

# Round monetary values
df["claim_amount"] = df["claim_amount"].round(2)

# Save CSV
output_file = "synthetic_health_claims.csv"
df.to_csv(output_file, index=False)

print("Dataset generated successfully!")
print("Total records:", len(df))
print("\nClass distribution:")
print(df["is_anomaly"].value_counts())
print("\nFirst five records:")
print(df.head())
print("\nSaved to:", output_file)

configuración del servidor MLflow y ejecución del experimento de Machine Learning (
entrenar y evaluar un modelo de Machine Learning.

Para este proyecto pueden usar un modelo denominado Isolation Forest.

registro del modelo y configuración de BentoML para servir modelos de Machine Learning
Informe con capturas y el paso a paso de la solución