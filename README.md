# XAI baseline vs explained model

Lane: ML / XAI
Use for: UvA explainable ML, general XAI PhDs, applied ML methods
Plan slot: 21 Oct to 8 Nov 2026 (primary flagship)
Status: wip (data, protocol and checks in place; no model results yet)

## What this project does
Predicts whether income exceeds 50K on the UCI Adult data and asks how far the explanations and the predictions can be trusted, with every analysis choice fixed in PROTOCOL.md before any model is fitted.

- Data: UCI Adult (Becker and Kohavi, 1996, CC BY 4.0), the original train and test files combined into 48,842 rows. A copy is committed in data/, so the analysis runs without network access; see data/SNAPSHOT.md for the source, licence, hash and handling of missing values.
- Evaluation: stratified 5-fold cross-validation repeated over 5 seeds (25 folds), with all preprocessing fitted inside each training fold.
- Models, with hyperparameters fixed in advance: LightGBM, and an L2-regularised logistic regression as an interpretable reference. A majority-class baseline sets the floor.
- Metrics: accuracy, F1 and ROC-AUC, as mean and standard deviation over the 25 folds.
- Explanations: interventional SHAP for both models on a shared background from the training fold, with one-hot attributions summed back to the original features.
- Stability check: agreement of feature importance rankings across folds and seeds (Kendall's tau and top-5 overlap).
- Faithfulness check: a deletion test that removes features in SHAP order and compares the drop in model confidence with random removal orders.
- Fairness check: demographic parity difference and equalized odds difference across sex and race for both models, with per-group rates and sizes, and the share of SHAP attribution falling on sex and race.
- Pre-specified secondary variant: both models retrained without sex and race, to test whether disparities persist through correlated features.

## Deliverables
- Reproducible pipeline: pinned requirements.txt, fixed seeds, committed data snapshot, tests
- PROTOCOL.md fixed before model fitting, with a change log
- Results tables and figures with a limits section in this README
- Two-page PDF note

## Running
- Install: `python -m pip install -r requirements.txt` (Python 3.12)
- Tests: `python -m pytest`
- Sanity outputs (class balance, group sizes, majority-class baseline): `python -m src.sanity`

## Links (fill when live)
- GitHub:
- PDF / note:
- Dataset source: https://archive.ics.uci.edu/dataset/2/adult

## Notes
Keep commits small and README honest about limits. Prefer one solid result over many unfinished notebooks.
