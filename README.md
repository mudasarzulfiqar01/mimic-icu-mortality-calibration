# ICU Mortality Prediction on MIMIC-IV: Calibration, Missing Data, Validation Design and Subgroup Reporting

**Team:** Mudassar, Touseef, Ifraz

Term project accompanying our systematic review *Calibration, Missing Data, Validation Design and Subgroup Reporting
in Machine-Learning Models for ICU Mortality Prediction on MIMIC Data*. The review audits how published MIMIC-based
ICU mortality models report calibration (RQ1), handle missing data (RQ2), validate without leakage (RQ3), compare with
baselines and published severity scores (RQ5) and report subgroup performance (RQ6). This repository contains the
review's **illustrative case study**: one fixed MIMIC-IV cohort, modelled with the practices the review recommends.

**Prediction task:** in-hospital mortality from the first 24 hours of the ICU stay.
**Cohort:** adults (age ≥ 18), first ICU stay per patient, ICU stay ≥ 24 h → 51,838 stays, 10.65% mortality.

| Part | Notebook | Content |
|---|---|---|
| **1. Data preparation + EDA** | [`1_eda/MIMIC_ICU_Mortality_EDA_GPU_executed.ipynb`](1_eda/MIMIC_ICU_Mortality_EDA_GPU_executed.ipynb) | cohort, first-24 h feature extraction, cleaning, feature engineering, EDA, patient-grouped split, train-only preprocessing, feature selection |
| **2. Modelling + evaluation** | [`2_model/MIMIC_ICU_Mortality_Modelling_executed.ipynb`](2_model/MIMIC_ICU_Mortality_Modelling_executed.ipynb) | train/validation/test split, 5 models + SOFA baseline, Optuna tuning, optimisation, calibration, evaluation with CIs, model comparison, visualisation, subgroup analysis, protocol experiments, final pipeline |
| 3. Report | `3_report/` | IEEE paper (in preparation) |

---

## Repository structure

```
README.md
requirements.txt
.gitignore
1_eda/
  MIMIC_ICU_Mortality_EDA_GPU_executed.ipynb   Part 1 (executed)
  figures/                                     16 EDA figures
2_model/
  MIMIC_ICU_Mortality_Modelling_executed.ipynb Part 2 (executed)
  figures/                                     modelling figures (ROC/PR, calibration, DCA, SHAP, subgroups, ...)
  results/                                     aggregate result tables (CSV/JSON)
3_report/                                      IEEE report
data/            NOT in the repository - MIMIC-IV files and processed data (credentialed)
```

> **Data are not included.** MIMIC-IV is credentialed data (PhysioNet data-use agreement). The repository contains only
> code, aggregate results and figures. Four Part 1 output cells that previewed individual patient rows were replaced
> with a note before publishing; re-run them locally to see them. The fitted model (`2_model/saved_models/`) is not
> published because it was trained on credentialed data.

---

## Key results (test set, 10,368 stays never used for any choice)

| Model | AUROC (95% CI) | AUPRC (95% CI) | Brier | ECE | Calibration slope |
|---|---|---|---|---|---|
| **Ensemble (XGBoost + LightGBM + MLP) — final** | **0.892 (0.884–0.900)** | **0.559 (0.527–0.585)** | 0.066 | 0.003 | 1.01 |
| XGBoost | 0.890 (0.882–0.898) | 0.534 (0.503–0.560) | 0.066 | 0.002 | 0.96 |
| LightGBM | 0.889 (0.881–0.897) | 0.533 (0.502–0.560) | 0.067 | 0.004 | 0.97 |
| MLP | 0.888 (0.879–0.896) | 0.520 (0.490–0.546) | 0.067 | 0.004 | 0.98 |
| Random Forest | 0.882 (0.872–0.891) | 0.521 (0.490–0.548) | 0.068 | 0.005 | 0.98 |
| Logistic regression (L2) | 0.869 (0.859–0.878) | 0.471 (0.443–0.498) | 0.071 | 0.006 | 0.96 |
| SOFA score (first 24 h) | 0.702 (0.687–0.717) | 0.229 (0.209–0.250) | 0.088 | 0.004 | 0.97 |

AUPRC baseline (prevalence) = 0.106. All models use isotonic recalibration fitted on the validation set.

| Protocol question | Finding |
|---|---|
| H3: ML vs. equally tuned logistic regression | XGBoost − LR: **+0.063 AUPRC** (paired bootstrap 95% CI 0.049–0.078); ECE difference not significant |
| RQ5: gain over a published score | LR − SOFA: +0.242 AUPRC; final model − SOFA: +0.330 AUPRC |
| RQ2: missing data | complete-case training keeps only 15% of stays and lowers AUPRC (LR 0.471 → 0.437, XGBoost 0.534 → 0.503); imputation with/without indicator and XGBoost native NaN handling perform alike |
| RQ3: temporal hold-out (2020–2022) | small AUROC drop (LR 0.871 → 0.851, XGBoost 0.890 → 0.879) |
| RQ6: subgroups | largest AUROC gap by age band (0.833–0.922); sex 0.888–0.895 |

Full tables are in `2_model/results/` (e.g. `test_model_comparison.csv`, `test_paired_bootstrap.csv`,
`test_subgroup_metrics.csv`, `protocol_missing_data_strategies.csv`, `protocol_temporal_holdout.csv`).

---

## Methods in brief

**Part 1 (data preparation).** First ICU stay of adults with ≥ 24 h stay; vitals, GCS, ICU-charted labs, urine output,
vasopressors and ventilation in the first 24 h (chunked reading of the 3.5 GB `chartevents` file); unit harmonisation
and removal of physiologically impossible values; mean/min/max/count features plus GCS total, shock index, PaO₂/FiO₂,
BUN/creatinine; patient-grouped stratified 80/20 split; log transform, winsorising, median imputation with
missing-value indicators, one-hot encoding and scaling **fitted on the training set only**; feature selection (near-zero
variance → correlation filter → mutual information + XGBoost importance) → 40 features.

**Part 2 (modelling).**
* **Split:** Part 1 training set split again by patient into train (64%) and validation (16%); test (20%) untouched until
  the final evaluation.
* **Models:** L2 logistic regression, Random Forest, XGBoost (GPU if available), LightGBM, MLP; SOFA computed from the
  raw first-24 h values as a published-score baseline.
* **Tuning:** Optuna (TPE) maximising grouped-CV AUPRC on the training set (3 folds, 6–25 trials per model).
* **Optimisation:** default vs. tuned, class weighting, 40 vs. 257 features, calibration (none / Platt / isotonic;
  isotonic pre-specified), soft-voting ensemble, decision thresholds (Youden, sensitivity ≥ 0.80) — all chosen on
  training/validation data only.
* **Evaluation:** AUROC, AUPRC, Brier, ECE (10 equal-frequency bins), calibration slope and intercept, sensitivity,
  specificity, PPV, F1 with 500-resample bootstrap 95% CIs; DeLong test; paired bootstrap of AUPRC/ECE differences;
  decision-curve analysis; SHAP; subgroup analysis; missing-data and temporal experiments.
* **Pipeline:** features → model(s) → validation-fitted calibrator → averaged risk → threshold, saved with `joblib` and
  verified to reproduce the test predictions after reloading.

---

## Data source

**MIMIC-IV v3.x**, PhysioNet: <https://physionet.org/content/mimiciv/>

> Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. *MIMIC-IV, a freely accessible electronic health record dataset.*
> Scientific Data 10, 1 (2023).

| File | Module | Used for |
|---|---|---|
| `patients.csv.gz` | hosp | sex, age |
| `admissions.csv.gz` | hosp | outcome (`hospital_expire_flag`), admission type, insurance, race |
| `icustays.csv.gz` | icu | ICU stays, care unit, admission time |
| `chartevents.csv.gz` | icu | vitals, GCS, labs charted in the ICU |
| `outputevents.csv.gz` | icu | urine output |
| `inputevents.csv.gz` | icu | vasopressor use |
| `procedureevents.csv.gz` | icu | mechanical ventilation |
| `d_items.csv.gz`, `d_labitems.csv.gz` | dictionaries | item ID → label lookups |

> **Access:** every user needs PhysioNet credentialing (CITI training and the signed data-use agreement). Do not
> share the data, commit it to Git, or upload it to public services.

---

## How to reproduce

**Hardware:** 16 GB RAM recommended (32 GB comfortable), ~5 GB free disk. A GPU is optional (XGBoost only).

1. **Get the data** from PhysioNet and place the `.csv.gz` files listed above in `data/`
   (or keep the archive at `E:\patients data.zip`; Part 1 extracts missing files automatically).
2. **Install the libraries** (Python 3.10+; the results were produced with Python 3.14):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Check the paths.** If your project folder is not `E:\patients data`, edit `PROJECT_DIR` (and `ZIP_PATH` in Part 1)
   in the paths cell of each notebook.
4. **Run Part 1, then Part 2**, top to bottom (VS Code / Jupyter → *Run All*), or from the command line:

   ```bash
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir 1_eda --output MIMIC_ICU_Mortality_EDA_run.ipynb 1_eda/MIMIC_ICU_Mortality_EDA_GPU_executed.ipynb
   ```

   ```bash
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir 2_model --output MIMIC_ICU_Mortality_Modelling_run.ipynb 2_model/MIMIC_ICU_Mortality_Modelling_executed.ipynb
   ```

   | Step | Time (8-core CPU, no GPU) |
   |---|---|
   | Part 1, first run (streams `chartevents`) | 15–30 min; later runs use the cache in `data/processed/` |
   | Part 2 | about 35 min (most of it hyperparameter tuning) |

   Part 1 writes `data/processed/` (`train_processed.parquet`, `test_processed.parquet`, `cohort_features_raw.parquet`,
   `feature_metadata.json`, `cohort_flow.csv`), which Part 2 reads.

5. **Quick smoke test of Part 2** (subsample, 2 trials per model, ~2 min): set the environment variable `ICU_FAST=1`
   before starting Jupyter. Results from this mode are not valid estimates.

**Kaggle:** both notebooks detect `/kaggle/input/` automatically. Because of the data-use agreement, attach the data only
as a **private** dataset you are allowed to use.

All random seeds are fixed at 42.

---

## Methodological safeguards

* One ICU stay per patient; every split is grouped by patient; no patient appears in two sets (asserted in code).
* All preprocessing, tuning, calibration, ensemble and threshold choices use training/validation data only.
* ICU length of stay, discharge information and date of death are never predictors.
* Calibration protocol fixed in advance (isotonic on validation, 10 equal-frequency bins, slope and intercept).
* Limitations: single centre; weak temporal check only (MIMIC-IV dates are shifted per patient); SOFA computed without
  vasopressor doses; 24-h window only.
