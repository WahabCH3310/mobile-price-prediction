# ══════════════════════════════════════════════════════════════════
#  Mobile Price Prediction — task runner
#  Usage:  make <target>   (run `make help` to list targets)
# ══════════════════════════════════════════════════════════════════
.PHONY: help install data eda train train-fast test api app clean all

PY := python3

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install all Python dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## Build the processed dataset (raw + proxy price)
	$(PY) -m src.data.make_dataset

eda:  ## Generate EDA figures -> reports/figures/
	$(PY) -m src.visualization.eda

train:  ## Full training run (all models + Optuna + SHAP)
	$(PY) -m src.models.train

train-fast:  ## Quick training run (fewer Optuna trials)
	$(PY) -m src.models.train --fast

test:  ## Run the unit tests
	$(PY) -m pytest -q

api:  ## Serve the FastAPI endpoint at :8000  (docs at /docs)
	uvicorn app.api:app --reload --port 8000

app:  ## Launch the Streamlit dashboard
	streamlit run app/streamlit_app.py

all: data eda train  ## Build data, run EDA, and train everything

clean:  ## Remove generated artifacts
	rm -rf models/*.joblib models/*.json data/processed/*.csv \
	       reports/figures/*.png reports/metrics.json
	@echo "cleaned generated artifacts."
