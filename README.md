# Dropout Lens

![ci](https://github.com/sahajm99/dropout-lens/actions/workflows/ci.yml/badge.svg)

Who leaves one Portuguese polytechnic, and what predicts it. The data are 4,424 students
who enrolled between 2008 and 2019, with an outcome of Dropout, Enrolled or Graduate
recorded at the normal end of the course. Every share on the site carries a 95% Wilson
interval, every hypothesis test carries an effect size and a Holm-adjusted p-value, and
every model is reported on two feature sets side by side, because the second one leaks
the outcome. A browser-side estimator puts a profile through the enrolment logistic model
and returns a probability with an interval.

Live: <https://sahajm99.github.io/dropout-lens/>

## The record, corrected

An earlier portfolio card for this coursework said "10,000+ student records" and "85%
accuracy on holdout set". The file has 4,424 rows and 34 features (the UCI page lists 36;
this copy lacks the two admission-grade columns). The Fall 2023 notebook's best figure
(random forest, 0.9146 accuracy) was dropout versus graduate on 3,630 rows with the
Enrolled class dropped, a random split, and both semesters' results as features. The
Spring 2024 notebook coded Enrolled together with Graduate.

Measured here, on a fixed stratified 80/20 split (3,539 train, 885 test):

| Dropout or not | Best model by CV | Test accuracy (95% bootstrap) | ROC AUC |
|---|---|---|---|
| Majority baseline | | 67.9% (65.0 to 71.2) | not defined |
| At enrolment | Gradient boosting | 75.7% (72.9 to 78.4) | 0.785 |
| After first semester (leaks) | Gradient boosting | 85.6% (83.4 to 88.1) | 0.893 |

The honest at-enrolment number is below the old card's 85%. Only the leaking feature set
reaches it.

## What the site shows

- **Who is in the data.** 32.1% dropped out (30.8 to 33.5), 17.9% were still enrolled and 49.9% graduated.
- **What differs at enrolment.** Outcome shares by gender, scholarship, tuition status, debtor status, age band, application route, course, attendance and displacement, with intervals and suppression under 30 students.
- **What the tests say.** 18 of 20 tests survive Holm adjustment, so the effect size is the finding: tuition fees (V 0.43), scholarship (0.30), course (0.24).
- **What a model can predict, and when.** Four models, two targets, two feature sets, five metrics with bootstrap intervals, and calibration curves.
- **What drives the model.** Odds ratios at enrolment: fees not up to date 14.5, male 1.79, age 40 and over 3.08, scholarship 0.31. Permutation importance puts tuition fees first at enrolment and first-semester units approved first after it.
- **Where it is unequal.** The enrolment model misses 56.6% of the women who left and 43.2% of the men; it wrongly flags 6.1% of women and 25.4% of men who stayed.
- **The estimator.** Eight inputs, a probability with a delta-method interval, and the fields held at reference stated on the panel.
- **What this cannot tell you.** One institution; Enrolled is ambiguous; first-semester features leak; tuition and debtor status are recorded during the course; no causal claims; small subgroups suppressed.

## Results

Test set of 885 students. Intervals are 95% percentile bootstrap over 1,000 resamples.
Rows marked "after first semester" leak the outcome.

| Target | Feature set | Model | Accuracy | Balanced accuracy | Macro F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|---|
| Dropout or not | At enrolment | Majority | 0.679 (0.650 to 0.712) | 0.500 | 0.404 | not defined | not defined |
| Dropout or not | At enrolment | Logistic | 0.779 (0.751 to 0.808) | 0.701 | 0.717 | 0.793 (0.759 to 0.829) | 0.714 |
| Dropout or not | At enrolment | Gradient boosting | 0.757 (0.729 to 0.784) | 0.689 | 0.700 | 0.785 (0.751 to 0.821) | 0.691 |
| Dropout or not | At enrolment | Random forest | 0.760 (0.732 to 0.789) | 0.706 | 0.713 | 0.778 (0.743 to 0.813) | 0.679 |
| Dropout or not | After first semester | Logistic | 0.861 (0.837 to 0.884) | 0.818 | 0.832 | 0.882 (0.856 to 0.910) | 0.840 |
| Dropout or not | After first semester | Gradient boosting | 0.856 (0.834 to 0.881) | 0.814 | 0.828 | 0.893 (0.867 to 0.919) | 0.859 |
| Dropout or not | After first semester | Random forest | 0.847 (0.823 to 0.872) | 0.810 | 0.819 | 0.892 (0.866 to 0.917) | 0.849 |
| Three-class | At enrolment | Majority | 0.499 (0.469 to 0.534) | 0.333 | 0.222 | not defined | not defined |
| Three-class | At enrolment | Logistic | 0.636 (0.603 to 0.671) | 0.521 | 0.510 | 0.763 | 0.610 |
| Three-class | At enrolment | Gradient boosting | 0.647 (0.614 to 0.681) | 0.547 | 0.550 | 0.765 | 0.613 |
| Three-class | At enrolment | Random forest | 0.620 (0.586 to 0.653) | 0.540 | 0.540 | 0.739 | 0.570 |
| Three-class | After first semester | Logistic | 0.723 (0.693 to 0.754) | 0.620 | 0.620 | 0.838 | 0.699 |
| Three-class | After first semester | Gradient boosting | 0.749 (0.721 to 0.779) | 0.676 | 0.681 | 0.865 | 0.729 |
| Three-class | After first semester | Random forest | 0.737 (0.705 to 0.764) | 0.654 | 0.657 | 0.863 | 0.721 |

The best model per row group is chosen by 5-fold cross-validated macro F1 on the training
set, never by the test set. The majority baseline is the same for both feature sets.

## Strongest effects

| Predictor | Cramér's V (corrected) | Holm p |
|---|---|---|
| First-semester approval rate (post-enrolment) | 0.474 | < 1e-300 |
| Tuition fees | 0.431 | 2.8e-178 |
| Scholarship | 0.304 | 1.2e-88 |
| Course | 0.244 | 3.2e-96 |
| Debtor status | 0.241 | 3.9e-56 |

Eta squared across the three outcomes: age at enrolment 0.065, first-semester grade 0.119.

## Fairness slice

Enrolment gradient boosting model, threshold 0.5, test set. FNR is the share of students
who left that the model missed; FPR is the share who stayed that it flagged.

| Group | Level | FNR (95% Wilson) | FPR (95% Wilson) |
|---|---|---|---|
| Gender | Female | 56.6% (48.4 to 64.3) | 6.1% (4.1 to 8.8) |
| Gender | Male | 43.2% (35.2 to 51.5) | 25.4% (19.7 to 32.0) |
| Scholarship | No scholarship | 46.6% (40.6 to 52.8) | 16.9% (13.6 to 20.8) |
| Scholarship | Scholarship holder | 77.4% (60.2 to 88.6) | 1.6% (0.5 to 4.6) |
| Age | 17 to 19 | 72.2% (61.4 to 80.8) | 4.4% (2.6 to 7.4) |
| Age | 30 to 39 | 26.4% (16.4 to 39.6) | 44.2% (30.4 to 58.9) |
| Tuition | Fees up to date | 73.3% (66.5 to 79.1) | 11.1% (8.8 to 13.9) |

A gap here is a property of this model on this data, not a finding about students.

## Run it

```bash
uv sync
uv run python -m pipeline      # writes site/public/data/*.json
uv run pytest                  # 43 tests
cd site
npm ci
npm run dev                    # http://localhost:5173/dropout-lens/
npm run typecheck
npm test                       # estimator parity: browser vs pipeline to 1e-6
npm run build
```

CI (`.github/workflows/ci.yml`) reruns the pipeline, compares the fresh JSON with the
committed JSON (`scripts/check_stale.py`, numeric tolerances for model files), runs
pytest, typecheck, the vitest parity test and the build, and deploys to GitHub Pages.

## Layout

```
pipeline/       load, stats, descriptive, hypothesis, features, metrics, models,
                fairness, estimator, claims, run
tests/          stats, loader, descriptive, hypothesis, metrics, models,
                fairness and estimator, structural checks over every JSON, golden file
site/           Vite + TypeScript + Plotly; src/charts/ holds one module per figure
data/raw/       dataset.csv and SOURCE.md
docs/           DESIGN, DECISIONS, DATA_AUDIT, SOURCES, PROGRESS
scripts/        check_stale.py
```

## Data, licence and citation

Code: MIT, Sahaj Mekala. Data: CC BY 4.0, from the UCI Machine Learning Repository,
*Predict Students' Dropout and Academic Success*, <https://doi.org/10.24432/C5MC89>.

Realinho, V.; Machado, J.; Baptista, L.; Martins, M. V. Predicting Student Dropout and
Academic Success. *Data* 2022, 7(11), 146. <https://doi.org/10.3390/data7110146>

Rebuilt from two UNT course projects on the same file: CSE 5210 Fundamentals of AI
(Fall 2023) and Empirical Analysis (Spring 2024).

Numbers in this README were copied from the committed JSON produced by the pipeline run
of 2026-09-13; the site reads the same files.
