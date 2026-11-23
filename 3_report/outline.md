# Writing guide for the IEEE report (Part 3)

The LaTeX file `main.tex` is complete except for the prose. Each red **[TEAM: …]** box in the PDF marks one paragraph
for you to write **in your own words**. This guide gives, per section, what the paragraph must say, the numbers to use
and where they come from, plus a suggested split of the work. Write from understanding, not by copying sentences from
the README, the notebooks, the protocol or papers. That keeps the Turnitin similarity below 15%, makes the AI-detection
report reflect your own writing, and prepares you for the defence.

## Target length and division of work

About 6–8 pages in IEEE two-column format (check whether your course sets a limit).

| Section | Words (approx.) | Suggested author |
|---|---|---|
| Abstract, I. Introduction | 200 + 450 | Mudassar (search/protocol lead) |
| II. Related Work | 400 | Mudassar |
| III. Dataset and Cohort | 300 | Touseef (data extraction) |
| IV. Methodology, V. Setup | 800 + 150 | Touseef |
| VI. Results | 800 | Ifraz (synthesis, tables, figures) |
| VII. Discussion, VIII. Limitations, IX. Conclusion | 450 + 150 + 120 | Ifraz |
| Final read-through | — | all three; each must be able to explain every section |

## Build and check

```bash
cd 3_report
python build_tables.py
latexmk -pdf -interaction=nonstopmode main.tex
```

Before submitting:

1. No red boxes left in `main.pdf`.
2. Ask Claude to run the **ieee-format-checker**, **results-checker** and **reference-verifier** agents and fix what
   they report.
3. Run Turnitin and confirm the similarity index is below 15%. If it isn't, rewrite the flagged passages from
   understanding and make sure every borrowed idea is cited. Don't use "paraphrasing" tools.

## Section by section

### Abstract (150–250 words, one paragraph, no citations)

Problem → what you did → key numbers → conclusion. Numbers:

- cohort: 51,838 stays, 10.65% mortality
- final ensemble: AUROC 0.892 (0.884–0.900), AUPRC 0.559 (0.527–0.585), ECE 0.003, calibration slope 1.01
- XGBoost − logistic regression: +0.062 AUPRC
- SOFA: AUROC 0.702
- largest subgroup gap: age, AUROC 0.833–0.922

### I. Introduction

1. **Why it matters:** early ICU mortality risk; the existing scores SOFA [vincent1996sofa] and SAPS II [legall1993saps].
2. **Gap:** most papers report AUROC only. Calibration [vancalster2019calibration], missing data, leakage and
   subgroups are rarely reported. Reporting guideline: TRIPOD+AI [collins2024tripodai]. Link to your systematic review.
3. **Aim and questions:** calibration, missing data, validation design, gain over logistic regression and SOFA,
   subgroups.
4. **Contributions:** an `itemize` list of 3–4 items (suggestions are in `main.tex`).

### II. Related Work

- Harutyunyan et al. [harutyunyan2019multitask]: MIMIC-III benchmark.
- Purushotham et al. [purushotham2018benchmarking]: deep learning benchmarks.
- Johnson et al. [johnson2017reproducibility]: results change with cohort definition.
- Christodoulou et al. [christodoulou2019systematic]: no average ML benefit over logistic regression.
- Calibration [vancalster2019calibration, guo2017calibration]; missing-indicator method [groenwold2012missing];
  bias [obermeyer2019dissecting].
- End with how your study differs.

Read at least the abstract of each paper before summarising it. Never state a number from a paper you have not checked.

### III. Dataset and Cohort

- **Source:** MIMIC-IV [johnson2023mimiciv], PhysioNet [goldberger2000physionet], credentialed access.
- **Cohort flow** (Table I, Fig. 1): 94,458 → 65,366 (first stay) → 51,838 (≥ 24 h).
  - 5,522 deaths (10.65%), class ratio 8.4:1.
- **Missing data** (Fig. 2):
  - bilirubin, AST, PaCO₂ and PaO₂ are missing in 52–56% of stays;
  - complete cases are only 13.5% of stays;
  - mortality is 16.9% when AST was measured vs. 5.6% when it was not.

### IV. Methodology

| Subsection | Key facts |
|---|---|
| A. Features and preprocessing | 12.5 M measurements, 11,874 impossible values removed; 257 → 40 features; preprocessing fitted on train only |
| B. Validation design | train 33,176 / validation 8,294 / test 10,368; grouped by patient; role of each set |
| C. Models | why each one (citations in `main.tex`); LR vs XGBoost is the pre-specified comparison (H3) |
| C. SOFA | six organ systems, two deviations (no vasopressor dose; missing component = 0) |
| D. Tuning | Optuna TPE, 3-fold grouped CV, objective AUPRC, 6–25 trials (Table II) |
| E. Calibration | isotonic, pre-specified, fitted on validation; Platt / none as sensitivity analysis; ensemble rule (≥ 0.005 AUPRC); Youden threshold |
| F. Metrics | the equations are already in `main.tex`; explain the symbols in words; 500 bootstrap resamples; DeLong; paired bootstrap; SHAP |
| G. Protocol experiments | four missing-data strategies; temporal hold-out 2008–2019 → 2020–2022 and why it is weak |

### V. Experimental Setup

- Software: Python 3.14, scikit-learn 1.8, XGBoost 3.4, LightGBM 4.7, Optuna 5.1.
- Hardware: 8-core CPU, 32 GB RAM, no GPU. Run time about 35 minutes. Seed 42.
- GitHub URL.

### VI. Results (all numbers are in `tables/` and `2_model/results/`)

**A. Overall** (Table III, Fig. 3):

| Model | AUROC | AUPRC |
|---|---|---|
| Ensemble | 0.892 | 0.559 |
| XGBoost | 0.890 | 0.534 |
| LightGBM | 0.889 | 0.533 |
| MLP | 0.888 | 0.520 |
| Random forest | 0.882 | 0.521 |
| Logistic regression | 0.869 | 0.471 |
| SOFA | 0.702 | 0.229 |

- The test-set prevalence is 0.106 (the AUPRC of a random guess).
- At the Youden threshold of 0.125: sensitivity 0.822, specificity 0.800, PPV 0.329.
- DeLong test: the ensemble's AUROC is significantly higher than every other model's.

**B. Calibration** (Table IV, Fig. 4):

- ECE ≤ 0.006 for every model; slopes 0.96–1.01.
- Final model: slope 1.014, intercept 0.011.
- Class weighting raised mean predicted risk above observed mortality (`val_class_weighting.csv`).

**C. ML vs. LR vs. SOFA** (Table V, Fig. 5), AUPRC differences:

| Comparison | Δ AUPRC (95% CI) |
|---|---|
| XGBoost − LR | +0.062 (0.049–0.078) |
| Ensemble − LR | +0.087 |
| LR − SOFA | +0.242 |
| Ensemble − SOFA | +0.330 |

None of the ECE differences is significant.

**D. Utility and explainability:**

- Decision-curve analysis (Fig. 6): describe the range of thresholds where the model's curve lies above "treat all".
- SHAP (Fig. 7): top features are urine output, BUN, respiratory rate, GCS components, BUN/creatinine ratio, INR and
  insurance = Medicare. Comment on whether each direction is clinically plausible.

**E. Subgroups** (Table VI, Fig. 8), AUROC range:

| Group | AUROC range |
|---|---|
| Age band | 0.833–0.922 (largest gap) |
| Insurance | 0.870–0.918 |
| Admission period | 0.873–0.918 |
| Sex | 0.888–0.895 |

**F. Missing data and time** (Table VII, Fig. 9, Table VIII):

- Complete-case training used 15% of stays and lowered AUPRC: LR 0.437 vs. 0.471; XGBoost 0.503 vs. 0.534.
- Only one of the 40 selected features is a missing indicator.
- Temporal hold-out, AUROC: LR 0.871 → 0.851; XGBoost 0.890 → 0.879.

### VII. Discussion

1. Main findings.
2. Literature comparison with caveats:
   - cohorts and prevalence differ, so AUPRC is not comparable across studies;
   - many higher published numbers come from record-level splits, longer windows or leakage.
3. H3 and calibration.
4. Missing data and fairness.

### VIII. Limitations

- single centre;
- weak temporal test;
- no external validation;
- SOFA computed without vasopressor doses;
- 24-hour window only;
- reduced tuning budget;
- coarse race and insurance categories.

### IX. Conclusion

3–5 sentences, plus future work and the GitHub link.

### Acknowledgment

- PhysioNet / MIT-LCP.
- Credentialing statement.
- AI-assistance statement if your course requires one. Be accurate: AI helped with code, tables, figures and the
  LaTeX template.

## Defence preparation: questions to rehearse

- Why group the split by patient, and what would go wrong otherwise?
- Why AUPRC as the tuning objective? What is its baseline?
- What do ECE, calibration slope and intercept mean? Why isotonic, and why fit it on validation data only?
- Why does class weighting hurt calibration?
- How was SOFA computed, and what are its limitations here?
- What does the paired bootstrap show that two separate CIs don't?
- Why did complete-case training perform worse?
- Why is the `anchor_year_group` temporal test weak?
- Which subgroup performs worst, and what would you do about it?
- How does the final pipeline make a prediction for a new patient?
