# Notebooks

Numbered Jupyter notebooks that walk through the project end-to-end. Each notebook is self-contained and can be re-run from a fresh repo (assuming `data/processed/` has been built — see `../data/README.md`).

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_data_collection.ipynb` | Demonstrates how `nba_api` is used to scrape player game logs and team defensive stats. Shows the rate-limiting + caching strategy on a small sample. |
| 02 | `02_eda.ipynb` | Exploratory data analysis: Game Score distributions, the underperform target balance, era effects (why we z-score by season), star-vs-role-player comparisons. |
| 03 | `03_feature_engineering.ipynb` | Constructs the final modeling table step by step, with leakage checks at each stage (asserts that no feature references future games). |
| 04 | `04_model_training.ipynb` | Trains Logistic Regression, Random Forest, and XGBoost; performs time-series cross-validation and Optuna hyperparameter tuning; produces SHAP plots and final evaluation metrics. |

## Running

```bash
pip install -r ../requirements.txt
jupyter notebook
```

All notebooks import from `src/` rather than redefining logic — this guarantees that what the notebook demonstrates is exactly what the API runs in production.
