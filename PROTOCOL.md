# Analysis protocol

Fixed on 2026-10-21, before any model other than the majority-class baseline was fitted. Every choice below has a one-line reason. Any later change goes in the change log at the end, with its date, whether it was made before or after model results, and why. No hyperparameter or design choice is tuned on held-out folds.

## Data
- Source: UCI Adult (Becker and Kohavi, 1996), CC BY 4.0, adult.data and adult.test combined: 48,842 rows. Frozen in data/adult.csv.gz; see data/SNAPSHOT.md. Reason: cross-validation over the full data uses every row, and the original split has no special status here.
- Target: income above 50K (1) or not (0). Reason: the standard task for this dataset.
- Features used (12): age, education_num, capital_gain, capital_loss, hours_per_week (numeric); workclass, marital_status, occupation, relationship, race, sex, native_country (categorical).
- Dropped: fnlwgt, because it is a census sampling weight describing how many people a row represents, not a property of the person; education, because it is a one-to-one relabelling of education_num.
- Missing values ("?" in workclass, occupation and native_country, 3,620 rows): kept as a category named "Unknown" in each column. Reason: dropping the rows would remove 7.4% of the data non-randomly, and imputing would invent values; an explicit category keeps every row and lets the models use missingness.
- Duplicate rows (52) are kept. Reason: they are part of the published data and removing them would be a choice the original data does not support.

## Evaluation
- Stratified 5-fold cross-validation on the income label, repeated for split seeds 0, 1, 2, 3 and 4: 25 folds. Each row is held out exactly once per seed. Reason: repeated stratified folds give a spread of estimates and enough folds to measure stability across both folds and seeds.
- All preprocessing (one-hot encoding, standardisation, deletion-test replacement values, SHAP background data) is fitted on the training part of each fold only and then applied to its held-out part. Reason: nothing learned from held-out rows may reach the model.
- Encoding: one-hot for every categorical feature, with categories learned from the training fold and unseen categories in the held-out fold encoded as all zeros. Both models use the same one-hot inputs; logistic regression also standardises the 5 numeric features. Reason: one shared feature space makes the two models' attributions directly comparable.

## Models
Hyperparameters are fixed in advance and not tuned.
- LightGBM classifier: 300 trees, learning rate 0.05, 31 leaves, minimum 20 rows per leaf, row subsampling 0.8 every iteration, feature subsampling 0.8, seed 42, deterministic mode, 1 thread. Reason: standard moderate settings that are not fitted to this data.
- Logistic regression: L2 penalty, C = 1.0, lbfgs solver, up to 5,000 iterations. Reason: the default strength of regularisation for an interpretable linear reference.
- Majority-class baseline: predicts the training fold's majority class with the training fold's positive rate as its score. Reason: the floor any model must beat.

## Metrics
- Accuracy, F1 for the positive class (income above 50K), and ROC-AUC on each held-out fold, with predictions at a threshold of 0.5 on the predicted probability. Reported as mean and standard deviation over the 25 folds. Reason: accuracy alone is misleading with 24% positives; F1 and ROC-AUC cover the positive class and ranking quality.

## Explanations
- SHAP values on every held-out fold, on the log-odds scale: TreeExplainer for LightGBM (tree path-dependent, the default with no background data); LinearExplainer for logistic regression, with the training fold as the background data. Reason: the standard exact explainer for each model class.
- Attributions for the one-hot columns of each categorical feature are summed back to the original feature, giving one value per row for each of the 12 features. Reason: importance should be stated for features people recognise, not for encoding columns.
- Global importance in a fold: the mean absolute SHAP value of each feature over the held-out rows.

## Stability check
- In each of the 25 folds, rank the 12 features by global importance. For every pair of folds (300 pairs), compute Kendall's tau-b between the two rankings and the top-5 overlap (the number of features in both top 5 lists, divided by 5). Report the mean, standard deviation and minimum over all pairs, separately for each model. Reason: tau measures agreement across the whole ranking; top-5 overlap measures agreement where a reader looks first.

## Faithfulness check (deletion test)
- In each fold, take 500 held-out rows sampled without replacement (sampling seed 1000 times the split seed plus the fold index).
- For each row, remove its features one at a time in order of decreasing absolute SHAP value for that row, replacing each removed feature with the training fold's median (numeric features) or most frequent value (categorical features). After each removal, record the model's predicted probability for the class it predicted with nothing removed. This gives a curve over 0 to 12 removed features.
- Comparison: the same procedure with 5 random removal orders per row (seeded), averaged.
- Report the area under each mean deletion curve, with the number of removed features scaled to the range 0 to 1 and integrated by the trapezoidal rule, for SHAP order and for random order, and their difference (random minus SHAP). Mean and standard deviation over 25 folds, for each model. Reason: if SHAP ranks the features the model relies on, removing them first should lower the model's confidence faster than removing features at random, giving a smaller area.

## Fairness
- Using fairlearn on each held-out fold, with predictions at threshold 0.5: demographic parity difference (largest minus smallest selection rate between groups) and equalized odds difference (the larger of the largest true positive rate gap and the largest false positive rate gap between groups), across sex and across race, for both models. Mean and standard deviation over 25 folds. Reason: the two most common group fairness measures, one on predictions alone and one conditional on the true label.
- Per-group selection rate, true positive rate, false positive rate, number of rows and number of positives are reported for each group.
- Small race groups: the primary race metrics use the race groups with at least 1,000 people in the full data (White, Black, Asian-Pac-Islander). Amer-Indian-Eskimo (470 people, 55 above 50K) and Other (406, 50 above 50K) have about 10 positives per held-out fold, too few for stable rates; they are reported descriptively, with rates computed per seed over all 5 held-out folds pooled (every row once) and then averaged over seeds. A sensitivity version of the race metrics over all five groups is also reported. Reason: differences driven by a handful of people would dominate a largest-minus-smallest metric.
- Sex and race are used only as recorded in the data (sex as Male or Female; five race categories).

## Attribution share
- In each fold, the share of total absolute SHAP (summed over all held-out rows and all 12 features) that falls on sex, and the share on race, for both models. Mean and standard deviation over 25 folds. Reason: shows how much each model's explanations attribute to the protected attributes directly.

## Pre-specified secondary variant: unaware models
- Both models retrained with sex and race removed from the inputs (10 features), with everything else identical. The same accuracy, F1, ROC-AUC and fairness metrics are reported, with sex and race still used to define groups at evaluation. Reason: tests whether disparity persists when the protected attributes are not inputs, through correlated features such as relationship (which includes Husband and Wife) and marital status. It cannot show which features carry the disparity, and it is not a fairness intervention.

## Sanity outputs
- Class balance, group sizes by sex and race, and the majority-class baseline over the 25 folds, produced after this protocol was fixed. The baseline learns only the training fold's class balance.

## Known limitations fixed in advance
- Adult comes from the 1994 US Census. Incomes, occupations and group compositions have changed since, and results do not describe any current population.
- The label is whether reported income exceeded 50K, not a decision anyone made about the person. Fairness metrics here describe how the models' errors and positive predictions differ by group; they do not measure harm from a real decision process.
- Sex and race are coarse, as recorded in the census; the categories do not capture identity, and Other and Amer-Indian-Eskimo are too small for stable estimates.
- SHAP explains the fitted models, not the world. The two explainers also differ: tree path-dependent SHAP and interventional linear SHAP make different assumptions about dependence between features, so attributions are comparable in scale and meaning only approximately.
- Fixed hyperparameters mean neither model is the best achievable; results compare two reasonable models, not tuned ones.
- Cross-validation folds come from one dataset, so the spread across folds understates the variation expected on new data.

## Software and seeds
- Python 3.12 with the pinned versions in requirements.txt, CPU only. Split seeds 0 to 4; model seed 42; deletion sampling seeds as above. Reason: every result must be reproducible from a clean clone.

## Change log
- 2026-10-21: protocol fixed.
