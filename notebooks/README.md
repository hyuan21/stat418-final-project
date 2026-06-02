# Notebooks

This directory contains exploratory Jupyter notebooks used during development.

| # | Notebook | Purpose |
|---|----------|---------|
| 02 | `02_eda.ipynb` | Exploratory data analysis: Game Score distributions, the underperform target balance, era effects (why we z-score by season), star-vs-role-player comparisons. |

The data-collection, feature-engineering, and model-training steps are implemented as production Python modules under `src/` rather than as notebooks, so they can be run from the command line as part of the deployment pipeline:

| Step | Script |
|------|--------|
| Data collection | `python -m src.data.collect_game_logs` |
| Parsing raw JSON → Parquet | `python -m src.data.parse_raw` |
| Feature engineering with leakage checks | `python -m src.data.build_features` |
| Model training + Optuna tuning | `python -m src.models.train` |
| Post-training analysis (SHAP, slice metrics) | `python scripts/analyze_model.py` |

This keeps the exact same logic that runs in CI and in the deployed API, instead of duplicating it across notebooks that might drift.

## Running the EDA notebook

```bash
pip install -r ../requirements.txt
jupyter notebook 02_eda.ipynb
```

The notebook imports from `src/` directly, so it stays in sync with the production code.
