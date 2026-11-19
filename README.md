# Práctica 4: Pipeline de Machine Learning de extremo a extremo

Servicio que carga reclamaciones de seguros, las analiza con un modelo Isolation Forest y señala los casos que requieren revisión adicional.

Grupo: Ricardo Acuña, César Pajuelo, Angello Soldi, Josué Velo, Paolo Medrano.

## Componentes

1. Generación de datos sintéticos: src/generate_data.py
2. Servidor MLflow y experimento (entrenamiento y evaluación): src/train.py
3. Registro del modelo (alias champion) e importación a BentoML: src/import_to_bentoml.py
4. Servicio de predicción con BentoML: src/service.py y bentofile.yaml
5. Informe con capturas: INFORME.md e INFORME.pdf
6. Presentación (publicada con GitHub Pages): presentacion/presentacion_practica4.html y presentacion/presentacion_practica4.pptx (se generan con presentacion/build.py)

## Ejecución

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5001 &
python src/train.py
python src/import_to_bentoml.py
bentoml serve src.service:ClaimsReviewService --port 3000 &
curl -X POST localhost:3000/upload_csv -F file=@data/new_claims.csv
```
