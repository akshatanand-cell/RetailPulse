FROM python:3.11-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python data/generate_data.py && \
    python notebooks/02a_data_validation.py && \
    python notebooks/02b_data_cleaning.py && \
    python notebooks/03_segmentation.py && \
    python notebooks/04_demand_forecasting.py && \
    python notebooks/05_churn_prediction.py && \
    python notebooks/06_inventory_optimization.py && \
    python notebooks/07_drift_detection.py

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
