# ── Mobile Price Prediction — container image ─────────────────────
# Builds an image that serves the Streamlit dashboard (default) and can also
# run the FastAPI endpoint.  Works on HuggingFace Spaces (Docker SDK), Render,
# Railway, or any container host.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Train models at build time if artifacts are absent (keeps the image runnable
# even without committed .joblib files).  Comment out if you commit artifacts.
RUN test -f models/best_classifier.joblib || python -m src.models.train --fast

EXPOSE 8501 8000

# Default: Streamlit dashboard.  Override CMD to run the API instead:
#   docker run -p 8000:8000 <image> uvicorn app.api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
