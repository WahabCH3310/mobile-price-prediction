# 📱 Mobile Price Prediction Using Machine Learning

> **Final Year Project** — Department of Computer Science
> University of Education, Lahore (Vehari Campus)
> **Authors:** CH Abdul Wahab · M. Usman Ali &nbsp;|&nbsp; **Supervisor:** Dr. Munawar Hussain

An end-to-end machine-learning system that predicts a smartphone's **price tier**
(Budget → Flagship) and an **estimated price** from its hardware specifications,
with transparent **SHAP** explanations and two deployment channels (an
interactive **Streamlit** dashboard and a **FastAPI** REST endpoint).

---

## ✨ Results at a glance

All acceptance targets from the proposal (§12.1) are **met** on a held-out 20 % test set:

| Task | Metric | Target | Achieved | Model |
|------|--------|:------:|:--------:|-------|
| Classification | Accuracy | ≥ 93 % | **94.75 %** | Logistic Regression |
| Classification | F1-macro | ≥ 0.92 | **0.947** | Logistic Regression |
| Classification | ROC-AUC (OvR) | ≥ 0.95 | **0.998** | Logistic Regression |
| Regression | R² | ≥ 0.90 | **0.927** | LightGBM |
| Regression | MAE | < $45 | **$32.17** | LightGBM |
| Regression | RMSE | < $70 | **$40.62** | LightGBM |
| Regression | MAPE | < 12 % | **5.9 %** | LightGBM |

### Classifier comparison (test set)

| Model | Accuracy | F1-macro | ROC-AUC | CV F1-macro |
|-------|:--------:|:--------:|:-------:|:-----------:|
| **Logistic Regression** | **0.9475** | **0.9472** | 0.9982 | 0.9436 |
| LightGBM | 0.9350 | 0.9350 | 0.9942 | 0.9144 |
| XGBoost | 0.9300 | 0.9298 | 0.9941 | 0.9111 |
| Random Forest | 0.9125 | 0.9122 | 0.9859 | 0.8850 |
| MLP (Neural Net) | 0.8975 | 0.8979 | 0.9852 | 0.8844 |
| SVM | 0.8750 | 0.8756 | 0.9804 | 0.8496 |

> 🔎 **Interesting finding.** Logistic Regression *beats* the gradient-boosting
> ensembles on this dataset. That is expected here: `ram` is almost linearly
> separable across price tiers (see the SHAP analysis), so a linear decision
> boundary is nearly optimal. The tuned tree models remain the best choice for
> the **regression** target, where the spec→price relationship is non-linear.

---

## 🚀 The Streamlit app now includes PKR pricing + Pakistani market data

`app/streamlit_app.py` was redesigned with a dark, animated theme (Space Grotesk /
JetBrains Mono type, gradient hero, animated probability & SHAP bars, count-up price
numbers) and two new data sources, both usable entirely within the existing Streamlit
Cloud deployment — no separate hosting needed:

- **Live USD → PKR conversion** (`src/services/currency.py`) — every prediction shows
  the estimated price in USD and in PKR, using a live hourly-cached exchange rate with
  a graceful fallback if the rate API is briefly unreachable.
- **Pakistani market comparison panel** (`src/services/market.py`) — shows real
  PriceOye.pk retail listings alongside the model's own prediction, kept visually
  separate so the two are never confused (see note below).

Theme colors live in `.streamlit/config.toml`; just push this repo the same way you
deployed it originally (`git add . && git commit && git push`) and Streamlit Cloud
picks up the change and redeploys automatically — nothing else to configure.

An optional FastAPI backend (`app/api.py`) and static HTML frontend
(`app/static/index.html`) are also included in this repo for anyone who later wants a
standalone web app outside Streamlit, but they're **not required** — the Streamlit app
is fully self-contained.

### Keeping the Pakistani market data fresh

The market panel reads a cached JSON file rather than scraping PriceOye.pk on every
page load (scraping live per-request is slow and can get you rate-limited). Two ways
to refresh it:

- **Automatic:** `.github/workflows/refresh_market_data.yml` runs the scraper every
  6 hours and commits the result straight to the repo — Streamlit Cloud redeploys on
  that push automatically. GitHub-hosted runners are sometimes blocked by anti-bot
  rules; if the job stops finding rows, run the scraper from your own machine instead.
- **Manual:** `python -m src.data.scrape_priceoye --max-pages 10`, then commit + push.

Until it's run at least once, the market panel shows an empty state explaining this
rather than fake data.

### Notes on the price estimate

The regression target (`price_usd`) is trained on the public Kaggle dataset and is a
**demonstration proxy**, not a live market price (see `config/config.yaml`). The PKR
figure shown is a live currency conversion *of that proxy*, not a scraped real price —
the market panel is what shows actual current Pakistani retail prices, kept
deliberately separate so the two are never confused.

---

## 🗂️ Project structure

```
mobile-price-prediction/
├── config/config.yaml          # single source of truth for paths & hyper-params
├── data/
│   ├── raw/mobile_price.csv     # Kaggle seed dataset (2,000 rows)
│   └── processed/               # generated: dataset.csv, X_test.csv, ...
├── notebooks/
│   ├── 01_eda.ipynb             # exploratory data analysis
│   └── 02_modeling.ipynb        # training, tuning & SHAP
├── src/
│   ├── config.py                # YAML config loader
│   ├── data/
│   │   ├── make_dataset.py      # load Kaggle + build proxy price target
│   │   ├── scrape_gsmarena.py   # GSMArena scraper (rate-limited, robots-aware)
│   │   └── scrape_priceoye.py   # PriceOye.pk live-price scraper
│   ├── features/build_features.py   # 5 engineered features
│   ├── models/
│   │   ├── preprocessing.py     # impute / encode / scale ColumnTransformer
│   │   ├── model_zoo.py         # 6 classifiers + 3 regressors
│   │   ├── tune.py              # Optuna Bayesian optimisation
│   │   ├── train.py             # full training orchestration
│   │   ├── evaluate.py          # metrics + diagnostic plots
│   │   ├── explain.py           # SHAP global importance
│   │   └── predict.py           # inference service (used by app + API)
│   └── visualization/eda.py     # EDA figure generation
├── app/
│   ├── streamlit_app.py         # interactive dashboard
│   └── api.py                   # FastAPI REST endpoint (+ Swagger)
├── models/                      # generated: trained .joblib artifacts
├── reports/
│   ├── figures/                 # generated: all EDA/eval/SHAP PNGs
│   └── metrics.json             # generated: full results
├── tests/test_pipeline.py       # unit tests (pytest)
├── requirements.txt · Makefile · Dockerfile · LICENSE
```

---

## 🚀 Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Build the dataset, run EDA, and train everything
make all            # == make data + make eda + make train

# 3. Launch the dashboard  (http://localhost:8501)
make app

# 4. …or serve the REST API (Swagger at http://localhost:8000/docs)
make api
```

Run `make help` to see every target. A quick training run (fewer Optuna trials)
is available via `make train-fast`.

---

## 🧱 Methodology (CRISP-DM)

1. **Data collection** — the public Kaggle *Mobile Price Classification*
   dataset (2,000 phones, 20 specs, balanced 4-class target) is the reproducible
   seed. Scrapers for **GSMArena** (full spec sheets) and **PriceOye.pk** (live
   PKR prices) are provided to enrich the dataset with real market prices — see
   *Data notes* below.
2. **EDA** — class balance, price distributions, spec↔price relationships and a
   correlation heatmap (`src/visualization/eda.py`).
3. **Preprocessing** — median imputation, one-hot encoding for categoricals, and
   `StandardScaler` for the distance/gradient models (Logistic Regression, SVM,
   MLP). SMOTE is wired in but not triggered — the Kaggle data is already
   balanced.
4. **Feature engineering** — five derived features (`src/features/build_features.py`):
   `screen_area`, `pixel_density`, `camera_score`, `battery_per_gram`,
   `is_flagship`. Two of these (`battery_per_gram`, `pixel_density`) rank in the
   SHAP top-6, confirming they add signal.
5. **Modeling** — six classifiers and three regressors trained with 5-fold
   stratified CV.
6. **Tuning** — Optuna (TPE) tunes XGBoost & LightGBM over the proposal's search
   space (§7.1).
7. **Evaluation** — accuracy, F1-macro, ROC-AUC, confusion matrix (classification);
   MAE, RMSE, R², MAPE (regression).
8. **Explainability** — SHAP global importance + per-prediction contribution
   breakdowns surfaced in the dashboard.

---

## 📊 Feature importance (SHAP)

Top price drivers identified by tree-SHAP on the tuned LightGBM model:

| Rank | Feature | mean \|SHAP\| |
|:----:|---------|:------------:|
| 1 | `ram` | 6.17 |
| 2 | `battery_power` | 1.34 |
| 3 | `px_width` | 1.02 |
| 4 | `px_height` | 0.91 |
| 5 | `battery_per_gram` *(engineered)* | 0.78 |
| 6 | `pixel_density` *(engineered)* | 0.29 |

This matches the proposal's hypothesis that **RAM is the dominant price driver**,
followed by battery and display resolution. Figures are saved to
`reports/figures/` (`09_shap_importance.png`, `10_shap_summary.png`).

---

## 🌐 Deployment

- **Streamlit dashboard** (`app/streamlit_app.py`) — enter specs, get the
  predicted tier, class probabilities, estimated price, and a live SHAP
  explanation. Deployable to **Streamlit Community Cloud** or **HuggingFace
  Spaces**.
- **FastAPI** (`app/api.py`) — `POST /predict` returns the full prediction as
  JSON with auto-generated Swagger docs at `/docs`.
- **Docker** — `docker build -t mobile-price . && docker run -p 8501:8501 mobile-price`
  serves the dashboard; override the command to run the API. Works on HuggingFace
  Spaces (Docker SDK), Render, or Railway.

Example API call:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"battery_power":3000,"clock_speed":2.0,"fc":8,"four_g":1,"int_memory":64,
       "m_dep":0.6,"mobile_wt":160,"n_cores":8,"pc":13,"px_height":1280,
       "px_width":720,"ram":3000,"sc_h":14,"sc_w":7,"talk_time":15,
       "three_g":1,"touch_screen":1,"wifi":1,"blue":1,"dual_sim":1}'
```

---

## 📁 Data notes (important)

- **Classification target** (`price_range`, 0–3) is the *real* Kaggle label.
- **Regression target** (`price_usd`) is a **demonstration proxy** generated in
  `make_dataset.py` — a specs-driven, noisy function of the real hardware
  columns. It exists so the regression pipeline runs end-to-end **before** the
  real scraped market prices are merged in. **Regression numbers therefore
  measure pipeline correctness, not real-world pricing accuracy.** Replace it
  with real GSMArena/PriceOye prices via `merge_scraped_prices()` before quoting
  regression results in the final report.
- The scrapers respect `robots.txt`, rate-limit to 1–2 s/request, collect only
  public product data, and identify themselves — per the proposal's ethics
  section (§13). They are intended to run on the authors' own machine.
- `brand_tier` from the proposal is not implemented on the Kaggle data (it has
  no `brand` column); it becomes available once the GSMArena `brand` field is
  merged. `screen_area` stands in as the fifth engineered feature until then.

---

## 🔁 Reproducibility

Every result is reproducible from a fixed seed (`config.yaml → project.random_seed = 42`):

```bash
make clean && make all          # regenerate data, figures, models, metrics
pytest -q                       # 8 unit tests
```

The Kaggle seed CSV can be re-fetched with `python -m src.data.make_dataset --download`.

---

## 🧰 Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · Optuna · SHAP ·
Matplotlib · Seaborn · Plotly · Streamlit · FastAPI · Docker · pytest

## 📜 License

MIT © 2026 CH Abdul Wahab, M. Usman Ali — University of Education, Lahore (Vehari).
