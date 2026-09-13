# Dropout Lens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the two UNT student-retention course projects as Dropout Lens: a Python pipeline that measures everything on the UCI dropout dataset and writes small JSON, and a static Vite + TypeScript + Plotly site with a browser-side estimator, live on GitHub Pages, with the portfolio card corrected.

**Architecture:** `python -m pipeline` reads `data/raw/dataset.csv` once, decodes the Table-1 codes into labelled ordered categoricals, and runs writers in order (descriptive, hypothesis tests, models, fairness, estimator, then claims and meta) into `site/public/data/*.json`, each under 200 KB. The site never reads a raw row: one chart module per figure fetches one JSON file, every number in the prose is a `data-stat` span filled from `claims.json`, and the estimator evaluates the enrolment logistic model's coefficients and covariance in the browser. CI reruns the pipeline, diffs the committed JSON with a numeric tolerance, runs pytest, typecheck, vitest parity, build and deploys Pages.

**Tech Stack:** Python 3.12 via uv (pandas 2.3, numpy 2.5, statsmodels 0.14.6, scikit-learn 1.9, scipy 1.18, pytest 8); Node 24, Vite 8, TypeScript 6, vitest 4, plotly.js-cartesian-dist-min 4; fonts `@fontsource-variable/literata` and `@fontsource-variable/bricolage-grotesque`.

**Spec:** `docs/DESIGN.md`, `docs/DECISIONS.md` (D1 to D8, S1 to S11, W1 to W10, R1 to R5), `docs/DATA_AUDIT.md`, `docs/SOURCES.md`. The mission brief is the approval gate.

## Global Constraints

- Python only via `uv run ...` (bare `python` is not on PATH). Node 24 and npm are installed. Windows shell; write every file with LF line endings and UTF-8.
- Every commit is authored as `sahajm99 <64627746+sahajm99@users.noreply.github.com>` (already set in the repo-local git config); no `Co-Authored-By` or any other trailer, ever.
- Source data `data/raw/dataset.csv` is read-only. Never modify anything under the UNT course folders.
- Every JSON written to `site/public/data/` is under 200,000 bytes, has `schema`, `id`, `generated_at` at the top level, is written with `allow_nan=False`, and every proportion satisfies `0 <= lo <= p <= hi <= 1` unless suppressed (then `p`, `lo`, `hi` are `null` and `suppressed` is `true`).
- Seed `20260912` for every random component; model fitting single-threaded (`OMP_NUM_THREADS=1` set in `pipeline/__init__.py` before scikit-learn is imported; `n_jobs=1` everywhere).
- Outcome order everywhere: `["Dropout", "Enrolled", "Graduate"]`. Outcome colours: Dropout `--series-2` (orange), Enrolled `--series-3` (aqua), Graduate `--series-1` (blue); never re-assigned.
- Suppression: a cell with `n < 30` is suppressed; `n < 100` is flagged `small_n`.
- No numeral describing the data is hand-typed in `site/index.html` or in a chart title: prose numbers are `<span data-stat="key|fmt">` filled from `claims.json`; chart titles are templates filled from the figure's own JSON. The only literal numerals allowed in `index.html` are CSS lengths, `viewBox` values and the year labels inside `data-stat` spans' fallbacks.
- The phrase "risk score" never appears anywhere in the repo's site, README or docs. The estimator copy contains, verbatim: "an association in one Portuguese institution's 2008 to 2019 records, not a prediction about a person and not a decision tool".
- The fairness section contains, verbatim: "A gap here is a property of this model on this data, not a finding about students."
- Every figure that uses the after-first-semester feature set has class `leaks` on its `<figure>` and the word "leaks" in its context line.
- Reference project for code shape: `C:/Users/sahaj/OneDrive/Desktop/Experiments/projects/active/cardiolens` (read-only). Copy its patterns; do not import from it.
- Test budget: under 70 pytest tests plus the vitest parity test. Tests only where a wrong number would otherwise go unnoticed.
- Implementers never dispatch subagents.

---

## File structure

```
dropout-lens/
  pyproject.toml, uv.lock, .python-version, .gitignore, LICENSE, README.md   (scaffolded)
  .github/workflows/ci.yml                     (scaffolded)
  scripts/check_stale.py                       (scaffolded)
  data/raw/dataset.csv, data/raw/SOURCE.md     (scaffolded)
  docs/DESIGN.md, DECISIONS.md, DATA_AUDIT.md, SOURCES.md, PROGRESS.md
  pipeline/
    __init__.py       sets OMP_NUM_THREADS=1 before anything imports sklearn
    __main__.py       from pipeline.run import run; run()
    constants.py      paths, seed, outcome order, code tables, level orders, group registry
    load.py           load_data() -> labelled DataFrame
    io.py             write_json(name, obj) -> Path (200 KB guard), now_iso()
    stats.py          wilson, cramers_v, eta_squared, epsilon_squared, holm, round6
    descriptive.py    outcome_overall.json, outcome_by_group.json
    hypothesis.py     hypothesis_tests.json
    features.py       feature sets, split, design helpers
    metrics.py        metric_suite, bootstrap_ci, calibration
    models.py         fit everything: model_metrics.json, calibration.json, odds_ratios.json, permutation_importance.json
    fairness.py       fairness.json
    estimator.py      estimator.json, estimator_parity.json
    claims.py         claims.json, meta.json (last)
    run.py            WRITERS list, clean_out_dir, run()
  tests/
    conftest.py, test_stats.py, test_load.py, test_descriptive.py, test_hypothesis.py,
    test_metrics.py, test_models.py, test_fairness_estimator.py, test_outputs.py,
    golden/outcome_overall.json
  site/
    package.json, tsconfig.json, vite.config.ts    (scaffolded)
    index.html
    public/.nojekyll, public/data/*.json
    src/main.ts, theme.ts, nav.ts, lazy.ts, data.ts, fmt.ts, figure.ts, stats-fill.ts,
        plotly.ts, plotly.d.ts, types.ts, hero.ts, estimator-core.ts, estimator-core.test.ts
    src/styles/tokens.css, base.css, layout.css, figure.css, panel.css
    src/charts/theme.ts, outcomeByGroup.ts, dropoutByGroup.ts, approvalBand.ts,
        effectSizes.ts, distributions.ts, metricsBinary.ts, metricsMulticlass.ts,
        calibration.ts, forest.ts, importance.ts, fairness.ts, estimator.ts
```

## JSON contracts (the seam between Python and TypeScript)

All files carry `schema`, `id`, `generated_at`. Proportions rounded to 6 decimals (`round6`), odds ratios and coefficients to 4, p-values to 3 significant digits (`sig3`), covariance entries to 10 decimals. Lists of levels are always in the categorical order defined in `pipeline/constants.py`.

### `outcome_overall.json` (`outcome_overall.v1`)
```json
{"schema":"outcome_overall.v1","id":"outcome_overall","generated_at":"...",
 "n_total":4424,"outcomes":["Dropout","Enrolled","Graduate"],
 "ci":{"method":"wilson","level":0.95},
 "rows":[{"outcome":"Dropout","n":1421,"p":0.321202,"lo":..,"hi":..}, ...],
 "tiles":{"Dropout":32,"Enrolled":18,"Graduate":50}}
```
`tiles` are largest-remainder integers summing to exactly 100.

### `outcome_by_group.json` (`outcome_by_group.v1`)
```json
{"schema":"outcome_by_group.v1","id":"outcome_by_group","generated_at":"...",
 "outcomes":[...],"ci":{"method":"wilson","level":0.95},"suppress_below":30,"small_n_below":100,
 "n_total":4424,
 "groups":[{"id":"gender","label":"Gender","post_enrolment":false,"levels":["Female","Male"],
   "rows":[{"level":"Female","n":2868,"counts":{"Dropout":720,"Enrolled":487,"Graduate":1661},
            "shares":{"Dropout":{"p":..,"lo":..,"hi":..},"Enrolled":{...},"Graduate":{...}},
            "suppressed":false,"small_n":false}]}]}
```
When suppressed, `shares` is `null`. Group ids and level orders (exactly):
- `gender` "Gender": Female, Male
- `scholarship` "Scholarship": "No scholarship", "Scholarship holder"
- `tuition` "Tuition fees": "Fees up to date", "Fees not up to date"
- `debtor` "Debtor status": "Not a debtor", "Debtor"
- `age_band` "Age at enrolment": "17 to 19", "20 to 22", "23 to 29", "30 to 39", "40 and over"
- `application_mode` "Application route": "1st phase, general contingent", "2nd or 3rd phase, general contingent", "Over 23 years old", "Transfer or change of course or institution", "Prior higher or technical qualification", "Special contingents and ordinances"
- `course` "Course": the 17 course labels ordered by n descending, ties by code
- `attendance` "Attendance": "Daytime", "Evening"
- `displaced` "Displaced": "Not displaced", "Displaced"
- `approval_band_s1` "First-semester approval rate" (`post_enrolment: true`): "No units enrolled", "0% approved", "1 to 49%", "50 to 99%", "100% approved"

### `hypothesis_tests.json` (`hypothesis_tests.v1`)
```json
{"schema":"hypothesis_tests.v1","id":"hypothesis_tests","generated_at":"...",
 "family_size":20,"adjustment":"holm","alpha":0.05,
 "categorical":[{"id":"gender","label":"Gender","post_enrolment":false,"test":"chi_square",
    "n":4424,"levels":2,"df":2,"statistic":..,"p_raw":..,"p_holm":..,
    "cramers_v":..,"cramers_v_corrected":..,"min_expected":..}],
 "numeric":[{"id":"age","label":"Age at enrolment","post_enrolment":false,"n":4424,"n_excluded":0,
    "anova":{"statistic":..,"df_between":2,"df_within":..,"p_raw":..,"p_holm":..,"eta_squared":..},
    "kruskal":{"statistic":..,"df":2,"p_raw":..,"p_holm":..,"epsilon_squared":..},
    "by_outcome":[{"outcome":"Dropout","n":..,"mean":..,"mean_lo":..,"mean_hi":..,"min":..,"q1":..,"median":..,"q3":..,"max":..}]}]}
```
Categorical ids (14): gender, scholarship, tuition, debtor, age_band, application_mode, course, attendance, displaced, marital ("Marital status": "Single", "Married or de facto union", "Divorced, separated or widowed"), previous_qualification ("Previous qualification": "Secondary", "Higher or post-secondary", "Basic or incomplete secondary"), international ("Nationality": "Portuguese", "International"), special_needs ("Educational special needs": "No special needs", "Special needs"), approval_band_s1 (post_enrolment). Numeric ids (3): age "Age at enrolment"; grade_s1 "First-semester grade" (post_enrolment, rows with grade 0 excluded); grade_s2 "Second-semester grade" (post_enrolment, same rule). Family size = 14 + 3 × 2 = 20; Holm across all 20.

### `model_metrics.json` (`model_metrics.v1`)
```json
{"schema":"model_metrics.v1","id":"model_metrics","generated_at":"...","seed":20260912,
 "split":{"n_train":3539,"n_test":885,"test_share":0.2,"stratified_on":"outcome"},
 "cv":{"folds":5,"selection_metric":"macro_f1"},"bootstrap":{"resamples":1000,"level":0.95},
 "feature_sets":{"enrolment":{"label":"At enrolment","leaks":false,"variables":[...labels...]},
                 "after_s1":{"label":"After first semester","leaks":true,"variables":[...]}},
 "targets":{"binary":{"label":"Dropout or not","classes":["Not dropout","Dropout"],"positive":"Dropout","test_positive_share":..},
            "multiclass":{"label":"Dropout, Enrolled or Graduate","classes":["Dropout","Enrolled","Graduate"]}},
 "models":["majority","logistic","hgb","rf"],
 "model_labels":{"majority":"Majority baseline","logistic":"Logistic regression","hgb":"Gradient boosting","rf":"Random forest"},
 "results":[{"target":"binary","feature_set":"enrolment","model":"hgb","leaks":false,
    "chosen_params":{"learning_rate":0.05,"max_iter":200} ,
    "cv":{"macro_f1_mean":..,"macro_f1_sd":..},
    "test":{"accuracy":{"value":..,"lo":..,"hi":..},"balanced_accuracy":{...},"macro_f1":{...},
            "roc_auc":{...},"pr_auc":{...}},
    "confusion":[[..,..],[..,..]],"classes":["Not dropout","Dropout"]}],
 "best":{"binary":{"enrolment":"hgb","after_s1":"hgb"},"multiclass":{"enrolment":"rf","after_s1":"hgb"}}}
```
For `majority`, `roc_auc` and `pr_auc` are `null` and `chosen_params` is `null`; for `logistic`, `chosen_params` is `null`. 16 results (2 targets × 2 feature sets × 4 models). `best` is the model with the highest `cv.macro_f1_mean` per (target, feature set), ties by model order.

### `calibration.json` (`calibration.v1`)
```json
{"schema":"calibration.v1","id":"calibration","generated_at":"...","bins":10,"strategy":"quantile","target":"binary",
 "curves":[{"feature_set":"enrolment","model":"logistic","leaks":false,"brier":..,
            "points":[{"mean_predicted":..,"fraction_positive":..,"n":..}]}]}
```
Six curves: logistic, hgb, rf × two feature sets.

### `odds_ratios.json` (`odds_ratios.v1`)
```json
{"schema":"odds_ratios.v1","id":"odds_ratios","generated_at":"...","fitted_on":"train",
 "models":{
   "binary_enrolment":{"label":"Dropout or not, at enrolment","target":"binary","feature_set":"enrolment","leaks":false,
      "covariates":["age_band","gender",...],"n_obs":3539,"n_events":..,"converged":true,"pseudo_r2_mcfadden":..,
      "intercept":..,"terms":[{"variable":"age_band","level":"17 to 19","reference":"17 to 19","label":"Age 17 to 19",
         "or":1.0,"lo":null,"hi":null,"coef":0.0,"se":null,"p_value":null,"n_level":..,"events_level":..,"is_reference":true,"unstable":false}, ...]},
   "binary_after_s1":{... same shape, leaks true, covariates plus "approval_band_s1"},
   "multiclass_enrolment":{"label":"Dropout, Enrolled or Graduate, at enrolment","target":"multiclass","feature_set":"enrolment","leaks":false,
      "base":"Graduate","covariates":[...],"n_obs":..,"converged":true,"pseudo_r2_mcfadden":..,
      "contrasts":{"Dropout":{"intercept":..,"terms":[...]},"Enrolled":{"intercept":..,"terms":[...]}}}}}
```
Term rows follow CardioLens `pipeline/models.py::_terms` exactly (one synthesised reference row per covariate first, then the fitted levels in design order). `unstable` is true when any cell of the level-vs-outcome 2x2 table (binary) is under 10.

### `permutation_importance.json` (`permutation_importance.v1`)
```json
{"schema":"permutation_importance.v1","id":"permutation_importance","generated_at":"...","repeats":10,"seed":20260912,
 "scoring":{"binary":"roc_auc","multiclass":"macro_f1"},
 "results":[{"target":"binary","feature_set":"enrolment","model":"hgb","leaks":false,"baseline_score":..,
             "features":[{"variable":"course","label":"Course","mean":..,"sd":..}]}]}
```
Four results (best model per target × feature set); features sorted by `mean` descending; a variable is permuted as a whole (the raw column, before encoding).

### `fairness.json` (`fairness.v1`)
```json
{"schema":"fairness.v1","id":"fairness","generated_at":"...","target":"binary","threshold":0.5,"ci":{"method":"wilson","level":0.95},"suppress_below":30,
 "results":[{"feature_set":"enrolment","model":"hgb","leaks":false,"n_test":885,
   "groups":[{"id":"gender","label":"Gender","levels":[{"level":"Female","n":..,"n_positive":..,"n_negative":..,
      "fnr":{"p":..,"lo":..,"hi":..},"fpr":{"p":..,"lo":..,"hi":..},"suppressed_fnr":false,"suppressed_fpr":false}]}]}]}
```
Groups: gender, scholarship, age_band, tuition. `fnr` (or `fpr`) is `null` and the matching `suppressed_*` true when its denominator is under 30.

### `estimator.json` (`estimator.v1`)
```json
{"schema":"estimator.v1","id":"estimator","generated_at":"...","model":"binary_enrolment","fitted_on":"train","z":1.959964,
 "intercept":..,"base_rate":..,"n_obs":3539,
 "inputs":[{"variable":"age_band","label":"Age at enrolment","levels":["17 to 19",...],"default":"17 to 19"},
           {"variable":"gender","label":"Gender","levels":["Female","Male"],"default":"Female"},
           {"variable":"scholarship","label":"Scholarship","levels":[...],"default":"No scholarship"},
           {"variable":"displaced","label":"Displaced","levels":[...],"default":"Not displaced"},
           {"variable":"application_mode","label":"Application route","levels":[...],"default":"1st phase, general contingent"},
           {"variable":"course","label":"Course","levels":[...],"default":"Nursing"},
           {"variable":"previous_qualification","label":"Previous qualification","levels":[...],"default":"Secondary"},
           {"variable":"marital","label":"Marital status","levels":[...],"default":"Single"}],
 "held_at_reference":[{"variable":"tuition","label":"Tuition fees","level":"Fees up to date"},
                      {"variable":"debtor","label":"Debtor status","level":"Not a debtor"},
                      {"variable":"international","label":"Nationality","level":"Portuguese"},
                      {"variable":"special_needs","label":"Educational special needs","level":"No special needs"}],
 "coefficients":{"age_band":{"17 to 19":0.0,"20 to 22":..},...},
 "terms":["Intercept","age_band=20 to 22", ...],
 "term_index":{"age_band":{"20 to 22":1,...},...},
 "cov":[[..]]}
```
`terms[0]` is `"Intercept"`; `term_index[variable][level]` gives the position of that non-reference level's term in `terms` and in `cov`; reference levels have no entry. The browser builds `x` (1 at index 0 and at each chosen non-reference level's index), computes `logit = sum(x_i * coef_i)`, `se = sqrt(x^T cov x)`, `p = sigmoid(logit)`, `lo = sigmoid(logit - z*se)`, `hi = sigmoid(logit + z*se)`.

### `estimator_parity.json` (`estimator_parity.v1`)
```json
{"schema":"estimator_parity.v1","id":"estimator_parity","generated_at":"...","seed":20260912,
 "rows":[{"inputs":{"age_band":"23 to 29","gender":"Male",...8 keys...},"p":..,"lo":..,"hi":..}]}
```
100 rows, values rounded to 10 decimals.

### `claims.json` (`claims.v1`)
Flat object of numbers and strings. Required keys (all derived from the other files or `meta`):
`n_rows, n_features, n_columns, year_from, year_to, n_dropout, n_enrolled, n_graduate, p_dropout, p_dropout_lo, p_dropout_hi, p_enrolled, p_graduate, tiles_dropout, tiles_enrolled, tiles_graduate, p_dropout_female, p_dropout_male, p_dropout_scholarship, p_dropout_no_scholarship, p_dropout_fees_late, p_dropout_fees_ok, p_dropout_debtor, p_dropout_not_debtor, p_dropout_age_youngest, p_dropout_age_oldest, p_dropout_evening, p_dropout_daytime, p_dropout_approval_0, p_dropout_approval_100, course_max_label, p_dropout_course_max, course_min_label, p_dropout_course_min, n_tests, n_tests_holm_sig, top_v_label, top_v, second_v_label, second_v, third_v_label, third_v, v_gender, v_age_band, v_approval_band_s1, eta_age, eta_grade_s1, n_train, n_test, test_positive_share, acc_majority_binary, acc_logistic_enrolment, acc_best_enrolment, acc_best_enrolment_lo, acc_best_enrolment_hi, best_enrolment_label, f1_best_enrolment, auc_best_enrolment, auc_best_enrolment_lo, auc_best_enrolment_hi, acc_best_after_s1, acc_best_after_s1_lo, acc_best_after_s1_hi, best_after_s1_label, f1_best_after_s1, auc_best_after_s1, acc_majority_multiclass, acc_best_multiclass_enrolment, best_multiclass_enrolment_label, f1_best_multiclass_enrolment, acc_best_multiclass_after_s1, best_multiclass_after_s1_label, f1_best_multiclass_after_s1, or_fees_late, or_debtor, or_scholarship, or_male, or_age_40, or_over23, fnr_female, fnr_male, fnr_gap_gender, fpr_female, fpr_male, top_importance_label_enrolment, top_importance_label_after_s1, estimator_base_rate`.
`course_max_label` etc. exclude suppressed rows. `n_tests_holm_sig` counts p_holm < 0.05 across the family. `v_*` are corrected Cramér's V. `or_age_40` is the binary enrolment OR for age "40 and over"; `or_over23` the application route "Over 23 years old". `fnr_*` from the enrolment best model.

### `meta.json` (`meta.v1`)
`schema, id, generated_at, source_file ("data/raw/dataset.csv"), sha256, n_rows, n_columns (35), n_features (34), n_duplicate_rows (0), outcome_counts {Dropout, Enrolled, Graduate}, seed, files (sorted list of every JSON name including meta.json)`. No library versions (they would differ by patch across platforms).

---

## Milestone 1: pipeline

### Task 1: Foundation: constants, load, io, stats

**Files:**
- Create: `pipeline/__init__.py`, `pipeline/__main__.py`, `pipeline/constants.py`, `pipeline/load.py`, `pipeline/io.py`, `pipeline/stats.py`
- Test: `tests/conftest.py`, `tests/test_stats.py`, `tests/test_load.py`

**Interfaces produced (every later task consumes these):**
- `pipeline.constants`: `ROOT`, `RAW_PATH`, `OUT_DIR`, `SEED = 20260912`, `OUTCOMES = ["Dropout","Enrolled","Graduate"]`, `YEAR_FROM = 2008`, `YEAR_TO = 2019`, `SUPPRESS_BELOW = 30`, `SMALL_N_BELOW = 100`, `Z95 = 1.959964`; `RENAME: dict[str,str]` raw column name to short name (see below); `LEVELS: dict[str, list[str]]` per derived categorical; `REFERENCE: dict[str, str]`; `GROUPS: list[dict]` registry `{"id","label","column","post_enrolment"}` in the order listed in the `outcome_by_group.json` contract plus `marital`, `previous_qualification`, `international`, `special_needs` (which the descriptive writer skips and the hypothesis writer uses); `APP_MODE_GROUP: dict[int,str]`, `COURSE_LABEL: dict[int,str]`, `MARITAL_GROUP: dict[int,str]`, `PREV_QUAL_GROUP: dict[int,str]`; `LABELS: dict[str, dict[str,str]]` human phrasing per (variable, level) for odds-ratio rows, e.g. `LABELS["age_band"]["40 and over"] = "Age 40 and over"`, `LABELS["tuition"]["Fees not up to date"] = "Tuition fees not up to date"`, `LABELS["course"][c] = f"Course: {c}"`; `VARIABLE_LABELS: dict[str,str]` variable to its group label.
- `pipeline.load.load_data() -> pd.DataFrame` with the raw columns renamed per `RENAME` and these derived columns (all `pd.Categorical(ordered=True)` with categories exactly `LEVELS[name]`): `outcome`, `gender`, `scholarship`, `tuition`, `debtor`, `displaced`, `international`, `special_needs`, `attendance`, `age_band`, `application_mode`, `course`, `marital`, `previous_qualification`, `approval_band_s1`; plus `dropout: int` (1 if outcome == "Dropout"). Raises `ValueError` if the raw column set is not the expected 35 or if any code is outside its table.
- `pipeline.io.write_json(name: str, obj: dict) -> Path` (copy CardioLens `pipeline/io.py`, `MAX_BYTES = 200_000`), `now_iso()`.
- `pipeline.stats`: `round6(x)`, `sig3(p)`, `wilson(events, n, z=Z95) -> (lo, hi)` (copy CardioLens), `share_cell(events, n) -> dict` returning `{"p","lo","hi"}` rounded or `None` when `n < SUPPRESS_BELOW`, `cramers_v(table: np.ndarray) -> tuple[float, float]` (plain, bias-corrected Bergsma 2013), `eta_squared(groups: list[np.ndarray]) -> float`, `epsilon_squared(h: float, n: int) -> float` (`h * (n + 1) / (n * n - 1)`), `holm(p: list[float]) -> list[float]` (via `statsmodels.stats.multitest.multipletests(method="holm")`), `mean_ci(x) -> (m, lo, hi)`.

`RENAME` (raw to short): `Marital status: marital_code`, `Application mode: app_mode_code`, `Application order: app_order`, `Course: course_code`, `Daytime/evening attendance: attendance_code`, `Previous qualification: prev_qual_code`, `Nacionality: nationality_code`, `Mother's qualification: mother_qual`, `Father's qualification: father_qual`, `Mother's occupation: mother_occ`, `Father's occupation: father_occ`, `Displaced: displaced_code`, `Educational special needs: special_needs_code`, `Debtor: debtor_code`, `Tuition fees up to date: tuition_code`, `Gender: gender_code`, `Scholarship holder: scholarship_code`, `Age at enrollment: age`, `International: international_code`, `Curricular units 1st sem (credited): s1_credited`, `... (enrolled): s1_enrolled`, `... (evaluations): s1_evaluations`, `... (approved): s1_approved`, `... (grade): s1_grade`, `... (without evaluations): s1_without_eval`, the same six for `2nd sem` as `s2_*`, `Unemployment rate: unemployment`, `Inflation rate: inflation`, `GDP: gdp`, `Target: outcome_raw`. Read with `encoding="utf-8-sig"`.

Code tables (from `docs/DATA_AUDIT.md` section 4, all verified):
- `COURSE_LABEL`: 1 Biofuel Production Technologies; 2 Animation and Multimedia Design; 3 Social Service (evening); 4 Agronomy; 5 Communication Design; 6 Veterinary Nursing; 7 Informatics Engineering; 8 Equinculture; 9 Management; 10 Social Service; 11 Tourism; 12 Nursing; 13 Oral Hygiene; 14 Advertising and Marketing Management; 15 Journalism and Communication; 16 Basic Education; 17 Management (evening). `LEVELS["course"]` is computed in `load.py` as labels sorted by count descending, ties by code ascending, and stored back on `constants.LEVELS["course"]` on first load so every module sees one order (`REFERENCE["course"] = "Nursing"`).
- `APP_MODE_GROUP`: {1} "1st phase, general contingent"; {8, 9} "2nd or 3rd phase, general contingent"; {12} "Over 23 years old"; {13, 14, 16, 18} "Transfer or change of course or institution"; {4, 15, 17} "Prior higher or technical qualification"; {2, 3, 5, 6, 7, 10, 11} "Special contingents and ordinances". Reference "1st phase, general contingent".
- `MARITAL_GROUP`: {1} "Single"; {2, 5} "Married or de facto union"; {3, 4, 6} "Divorced, separated or widowed". Reference "Single".
- `PREV_QUAL_GROUP`: {1} "Secondary"; {2, 3, 4, 5, 6, 14, 15, 16, 17} "Higher or post-secondary"; {7, 8, 9, 10, 11, 12, 13} "Basic or incomplete secondary". Reference "Secondary".
- Binary labels: gender 0 Female / 1 Male (ref Female); scholarship 0 "No scholarship" / 1 "Scholarship holder"; tuition 1 "Fees up to date" / 0 "Fees not up to date" (ref up to date); debtor 0 "Not a debtor" / 1 "Debtor"; displaced 0 "Not displaced" / 1 "Displaced"; international 0 "Portuguese" / 1 "International"; special_needs 0 "No special needs" / 1 "Special needs"; attendance 1 "Daytime" / 0 "Evening" (ref Daytime).
- `age_band`: `pd.cut(age, [16, 19, 22, 29, 39, 200], labels=LEVELS["age_band"])`; reference "17 to 19".
- `approval_band_s1`: `s1_enrolled == 0` "No units enrolled"; else rate = `s1_approved / s1_enrolled`: `rate == 0` "0% approved"; `0 < rate < 0.5` "1 to 49%"; `0.5 <= rate < 1` "50 to 99%"; `rate >= 1` "100% approved". Reference "100% approved".

- [ ] **Step 1: Write `tests/test_stats.py`** with these tests (exact expectations):
  - `wilson(0, 10)` gives `lo == 0.0` and `hi` within 1e-3 of 0.2775; `wilson(5, 10)` within 1e-3 of (0.2366, 0.7634); `wilson(10, 10)` gives `hi == 1.0`, `lo` within 1e-3 of 0.7225; `wilson(1, 0)` and `wilson(3, 2)` raise `ValueError`.
  - `cramers_v(np.array([[20, 10], [10, 20]]))` returns plain V within 1e-6 of `1/3` and corrected V within 1e-3 of 0.3095.
  - `eta_squared([np.array([1,2,3]), np.array([2,3,4]), np.array([3,4,5])])` equals 0.5 within 1e-9.
  - `holm([0.01, 0.04, 0.03])` equals `[0.03, 0.06, 0.06]` within 1e-9.
  - `share_cell(3, 29)` is `None`; `share_cell(15, 30)` has `lo <= p <= hi` and `p == 0.5`.
- [ ] **Step 2: Write `tests/test_load.py`** (`df` fixture in `tests/conftest.py`, session scoped, `load_data()`):
  - `len(df) == 4424` and `df["outcome"].value_counts().to_dict() == {"Graduate": 2209, "Dropout": 1421, "Enrolled": 794}`.
  - Every evening row is in a course whose label ends with "(evening)" and vice versa: `set(df.loc[df.attendance == "Evening", "course"].unique()) == {"Social Service (evening)", "Management (evening)"}` and no daytime row is in either.
  - Every "Over 23 years old" row has `age >= 23`; `df["age_band"].cat.categories.tolist() == LEVELS["age_band"]`; each derived categorical has no NaN.
  - `approval_band_s1` counts: "No units enrolled" 180, "0% approved" 538, "1 to 49%" 310, "50 to 99%" 1668, "100% approved" 1728.
  - `LEVELS["course"][0] == "Nursing"` and `len(LEVELS["course"]) == 17`.
- [ ] **Step 3: Run tests, confirm they fail** (`uv run pytest tests/test_stats.py tests/test_load.py -q`).
- [ ] **Step 4: Implement** `constants.py`, `load.py`, `io.py`, `stats.py`, `__init__.py` (`import os; os.environ.setdefault("OMP_NUM_THREADS", "1")`), `__main__.py`.
- [ ] **Step 5: Run tests, confirm they pass.**
- [ ] **Step 6: Commit** `feat(pipeline): constants, loader, stats helpers`.

### Task 2: Descriptive writers

**Files:** Create `pipeline/descriptive.py`, `tests/test_descriptive.py`.
**Consumes:** Task 1 interfaces. **Produces:** `write_outcome_overall(df) -> Path`, `write_outcome_by_group(df) -> Path`, and pure builders `build_outcome_overall(df) -> dict`, `build_outcome_by_group(df) -> dict` per the contracts. `tiles` via largest remainder: `floor(100*p)` each, then hand the remaining units to the largest fractional parts (ties by outcome order). Crosstabs with `observed=False` so every level appears even with zero rows.

- [ ] Tests: `build_outcome_overall(df)["tiles"]` sums to 100 and equals `{"Dropout": 32, "Enrolled": 18, "Graduate": 50}`; in `build_outcome_by_group`, the `course` group has exactly one suppressed row ("Biofuel Production Technologies", n 12) and row counts sum to 4424 for every group; every non-suppressed cell has `0 <= lo <= p <= hi <= 1` and the three outcome `p` values sum to 1 within 1e-6.
- [ ] Implement; run `uv run pytest tests/test_descriptive.py -q`; commit `feat(pipeline): outcome shares with Wilson intervals`.

### Task 3: Hypothesis tests writer

**Files:** Create `pipeline/hypothesis.py`, `tests/test_hypothesis.py`.
**Consumes:** Task 1. **Produces:** `build_hypothesis_tests(df) -> dict`, `write_hypothesis_tests(df) -> Path`.

Chi-square: `scipy.stats.chi2_contingency(table, correction=False)` on the `pd.crosstab(df[col], df["outcome"])` with `observed=False` then dropping all-zero rows; `min_expected` from the expected table. Numeric: age over all rows; `s1_grade` and `s2_grade` over rows with grade > 0 (`n_excluded` = rows dropped); ANOVA `scipy.stats.f_oneway`, eta squared from `stats.eta_squared`; Kruskal `scipy.stats.kruskal`, epsilon squared from `stats.epsilon_squared`; `by_outcome` summaries with `mean_ci`. Holm over the 20 raw p-values in the order categorical then numeric (anova, kruskal per variable); write back `p_holm`.

- [ ] Tests: `family_size == 20` and every `p_holm >= p_raw`; the `approval_band_s1` entry has the largest `cramers_v_corrected` of the categorical list; `gender` has `df == 2` and `levels == 2`; `grade_s1.n_excluded == 718`.
- [ ] Implement; run; commit `feat(pipeline): chi-square, ANOVA, Kruskal with effect sizes and Holm`.

### Task 4: Features, metrics and models

**Files:** Create `pipeline/features.py`, `pipeline/metrics.py`, `pipeline/models.py`, `tests/test_metrics.py`, `tests/test_models.py`.
**Consumes:** Task 1. **Produces:**
- `features.py`: `CAT_ENROLMENT = ["age_band","gender","scholarship","tuition","debtor","displaced","international","special_needs","application_mode","course","marital","previous_qualification"]` (the logistic covariates; attendance excluded per R3), `NUM_ENROLMENT = ["age","app_order","unemployment","inflation","gdp"]`, `TREE_EXTRA_ENROLMENT = ["attendance"]`, `CAT_AFTER_S1 = CAT_ENROLMENT + ["approval_band_s1"]`, `NUM_AFTER_S1 = NUM_ENROLMENT + ["s1_credited","s1_enrolled","s1_evaluations","s1_approved","s1_grade","s1_without_eval"]`; `FEATURE_SETS = {"enrolment": {"label":"At enrolment","leaks":False,"cat":CAT_ENROLMENT,"num":NUM_ENROLMENT,"tree_cat":CAT_ENROLMENT+["attendance"]}, "after_s1": {...}}`; `split(df) -> (train_idx, test_idx)` via `sklearn.model_selection.train_test_split(..., test_size=0.2, stratify=df["outcome"], random_state=SEED)` returning positional index arrays sorted ascending; `formula(target: str, cats: list[str]) -> str` building `f"{target} ~ " + " + ".join(f"C({c}, Treatment('{REFERENCE[c]}'))" ...)`.
- `metrics.py`: `metric_suite(y_true: np.ndarray, proba: np.ndarray, classes: list) -> dict[str, float|None]` with keys `accuracy, balanced_accuracy, macro_f1, roc_auc, pr_auc` (binary: `proba` is the positive-class column; multiclass: 2-D, `roc_auc_score(multi_class="ovr", average="macro")`, `pr_auc` = mean of per-class `average_precision_score`); `bootstrap_ci(y_true, proba, classes, n=1000, seed=SEED) -> dict[str, dict]` with `{"value","lo","hi"}` per metric (value from the full test set, percentile 2.5/97.5 over resamples drawn with `np.random.default_rng(seed).integers`); `calibration(y_true, p, bins=10) -> dict` with `points` and `brier`; `confusion(y_true, y_pred, classes) -> list[list[int]]`.
- `models.py`: `fit_all(df) -> FitBundle` (a dataclass holding, per (target, feature_set, model), the fitted predictor as a `Predictor` with `.predict_proba(frame: pd.DataFrame) -> np.ndarray` (n × k, columns in `classes` order), `.classes`, `.label`, `.chosen_params`, `.cv_macro_f1_mean/sd`; plus `train`, `test` DataFrames, `best: dict`, and the statsmodels results for `binary_enrolment`, `binary_after_s1`, `multiclass_enrolment`); `build_model_metrics(bundle) -> dict`, `build_calibration(bundle) -> dict`, `build_odds_ratios(bundle) -> dict`, `build_permutation_importance(bundle) -> dict`, and `write_models(df) -> list[Path]` writing the four files. `fit_all` is memoised on the module (`_BUNDLE`) so `fairness.py` and `estimator.py` reuse it inside one run.

Model details:
- Majority: predicts the training majority class; `predict_proba` returns one-hot of that class.
- Logistic: statsmodels `smf.logit(formula("dropout", cats), train)` and `smf.mnlogit(formula("outcome_code", cats), train)` where `outcome_code` is an int column 0 Graduate (base), 1 Dropout, 2 Enrolled (so Graduate is the base); fit `method="newton", maxiter=300, disp=0`, on non-convergence retry `method="bfgs", maxiter=3000`, raise `RuntimeError` if still not converged. CV macro F1: manual `StratifiedKFold(5, shuffle=True, random_state=SEED)` loop on train refitting the formula each fold.
- HGB and RF: `Pipeline([("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), tree_cat), ("num", "passthrough", num)])), ("model", ...)])`; `GridSearchCV(scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1)` with grids from DECISIONS S7 (HGB `model__learning_rate: [0.05, 0.1]`, `model__max_iter: [100, 200]`, fixed `max_depth=4, random_state=SEED`; RF `model__max_depth: [None, 8]`, `model__min_samples_leaf: [1, 5]`, fixed `n_estimators=300, random_state=SEED, n_jobs=1`). `chosen_params` are the winning grid values with the `model__` prefix stripped. CV mean/sd from `cv_results_` for the winner.
- Test metrics: `bootstrap_ci` on the test rows; `confusion` at argmax (binary: proba >= 0.5).
- Odds ratios: `_terms` per CardioLens with `LABELS`; multiclass contrasts parsed from `res.params` columns (statsmodels MNLogit columns are 0-based contrast indices in `classes[1:]` order: column 0 is Dropout vs Graduate, column 1 is Enrolled vs Graduate) with `conf_int()`.
- Permutation importance: manual: for the best model per (target, feature set), baseline score on test (`roc_auc_score` for binary on positive proba; `f1_macro` at argmax for multiclass); for each raw variable in `tree_cat + num` (or `cats` if the best model is logistic), 10 repeats: copy the test frame, permute that column with `np.random.default_rng(SEED)` (one generator for the whole run), score, importance = baseline minus permuted; emit mean and sd.
- Reproducibility: `fit_all` on the same data twice yields identical `model_metrics` (asserted once in a test via two runs of `build_model_metrics(fit_all(df))` on `bootstrap_ci` only, to keep the test fast: the bootstrap determinism test calls `bootstrap_ci` twice on a small synthetic array and asserts equality).

- [ ] Tests (`test_metrics.py`): `metric_suite(np.array([0,0,1,1]), np.array([0.1,0.4,0.35,0.8]), [0,1])` returns accuracy 0.75, balanced_accuracy 0.75, macro_f1 within 1e-6 of 0.733333, roc_auc 0.75, pr_auc within 1e-6 of 0.833333 (predictions at 0.5: [0,0,0,1] gives accuracy 0.75? No: [0,0,0,1] against [0,0,1,1] is accuracy 0.75, balanced 0.75, macro F1 = mean(0.8, 0.6667) = 0.733333. Use these numbers.); `bootstrap_ci` called twice with the same seed returns identical dicts and `lo <= value <= hi`; `calibration` points have `n` summing to `len(y)`.
- [ ] Tests (`test_models.py`, using the session `df` fixture and a module-scoped `bundle = fit_all(df)`): the split is stratified (test outcome shares within 0.01 of the full shares and `n_test == 885`); in `build_odds_ratios(bundle)` every model and contrast has exactly one `is_reference` row per covariate with `or == 1.0` and `level == reference`, and the number of non-reference rows per covariate equals `len(LEVELS[cov]) - 1`; `build_model_metrics(bundle)` has 16 results, every `lo <= value <= hi`, majority accuracy for binary within 1e-6 of the test not-dropout share, and every non-majority model beats the majority's balanced accuracy; `best` names a model with the top `cv.macro_f1_mean`.
- [ ] Implement; run `uv run pytest tests/test_metrics.py tests/test_models.py -q` (a few minutes); commit `feat(pipeline): models, metrics, odds ratios, importance`.

### Task 5: Fairness and estimator writers

**Files:** Create `pipeline/fairness.py`, `pipeline/estimator.py`, `tests/test_fairness_estimator.py`.
**Consumes:** Task 4 `fit_all`, `FitBundle`, `Predictor`; Task 1. **Produces:** `build_fairness(bundle) -> dict`, `write_fairness(df) -> Path`; `build_estimator(bundle) -> dict`, `build_parity(estimator: dict, seed=SEED, n=100) -> dict`, `write_estimator(df) -> list[Path]`; also `estimator.evaluate(est: dict, chosen: dict[str,str]) -> tuple[float,float,float]` (the Python twin of the browser function, used for parity rows).

Fairness: for each feature set, the `best["binary"][fs]` predictor; test rows; predicted positive when `proba[:, Dropout] >= 0.5`; per group level: `n`, `n_positive` (actual dropouts), `n_negative`; `fnr = FN / n_positive` with Wilson, suppressed when `n_positive < 30`; `fpr = FP / n_negative` likewise.

Estimator: from `bundle.binary_enrolment` (statsmodels result): `terms` in `res.params.index` order with `"Intercept"` first, parsed with the CardioLens `TERM_RE` into `f"{variable}={level}"`; `coefficients[variable][level]` for all levels (reference 0.0); `cov = res.cov_params()` rounded to 10 decimals, same order; `inputs` and `held_at_reference` exactly as the contract (labels from `VARIABLE_LABELS`); `base_rate` = train dropout share. `evaluate` builds `x`, computes `logit`, `se`, `p`, `lo`, `hi`. `build_parity` draws 100 profiles with `np.random.default_rng(seed)` choosing a uniform random level per input, evaluates each, rounds to 10 decimals.

- [ ] Tests: `build_fairness` has two results, groups in order gender, scholarship, age_band, tuition, every non-null rate has `lo <= p <= hi`, and `n` per group sums to 885; `build_estimator` has `terms[0] == "Intercept"`, `len(cov) == len(terms)`, `cov` symmetric within 1e-9, every input level except the reference has a `term_index` entry, and `evaluate(est, {defaults})` equals `sigmoid(intercept)` within 1e-9 (all defaults are references); `build_parity` has 100 rows with `lo <= p <= hi`.
- [ ] Implement; run; commit `feat(pipeline): fairness slice and browser estimator tables`.

### Task 6: Claims, meta, run, structural tests, golden

**Files:** Create `pipeline/claims.py`, `pipeline/run.py`, `tests/test_outputs.py`, `tests/golden/outcome_overall.json`.
**Consumes:** every writer. **Produces:** `site/public/data/*.json` (11 files: outcome_overall, outcome_by_group, hypothesis_tests, model_metrics, calibration, odds_ratios, permutation_importance, fairness, estimator, estimator_parity, claims, meta = 12 files), `python -m pipeline` working end to end.

`run.py` mirrors CardioLens (`clean_out_dir`, `WRITERS` in order: descriptive two, hypothesis, models, fairness, estimator, claims-and-meta last). `claims.py` mirrors CardioLens `build_claims(df, outputs)` reading the written files back; every key in the `claims.json` contract, derived (never recomputed) from those files; `build_meta` with the streaming sha256, `n_columns` from the raw header (35), `n_features` 34.

- [ ] Run `uv run python -m pipeline` and confirm 12 files, each under 200 KB, and that a second run changes only `generated_at` (`uv run python scripts/check_stale.py` cannot run before the first commit; instead diff two runs with the tolerance function imported from the script).
- [ ] `tests/test_outputs.py`: one parametrised structural test over every file in `site/public/data` (parses, `< 200_000` bytes, has `schema`/`id`/`generated_at`, no NaN token in the text, and if the file has `rows`/`groups`/`results` every `p/lo/hi` triple is ordered or null); one anchor test `meta.n_rows == 4424 and meta.n_features == 34 and claims.n_rows == 4424`; one anchor test that `claims.acc_best_enrolment` lies inside `[claims.acc_best_enrolment_lo, claims.acc_best_enrolment_hi]` and `claims.acc_best_after_s1 >= claims.acc_best_enrolment` (the leaking set cannot be worse on this data; if it is, the test fails and the controller rules); one golden test comparing `site/public/data/outcome_overall.json` minus `generated_at` to `tests/golden/outcome_overall.json` (copy the generated file to create the golden the first time).
- [ ] Run the full suite `uv run pytest -q` and record the count (must be under 70). Commit `feat(pipeline): claims, meta, run entry and structural tests` including the generated JSON and the golden file.

## Milestone 2: site chrome (can run in parallel with Milestone 1 tasks 2 to 6)

### Task 7: Site chrome, identity, hero tiles, copy

**Files:** Create `site/index.html`, `site/src/main.ts`, `theme.ts`, `nav.ts`, `lazy.ts`, `data.ts`, `fmt.ts`, `figure.ts`, `stats-fill.ts`, `plotly.ts`, `plotly.d.ts`, `types.ts`, `hero.ts`, `site/src/styles/{tokens,base,layout,figure,panel}.css`, `site/src/charts/theme.ts`.
**Consumes:** the JSON contracts above (types.ts declares every shape), `docs/DESIGN.md`, `docs/DECISIONS.md`, `docs/SOURCES.md`, `docs/DATA_AUDIT.md`. **Produces:** the module APIs every chart module uses, identical in signature to CardioLens: `figure.ts` (`FigureSpec`, `el`, `mountFigure`, `controlSlot`, `updateFigure`, `showError`), `data.ts` (`loadJson<T>(name)`), `fmt.ts` (`pct`, `pct0`, `ratio`, `int`, `ciText`, `oneIn`, `lowerLead`, `upperLead`, plus `num(x, digits)` and `pval(p)` which renders `< 0.001` below that), `charts/theme.ts` (`cssVar`, `baseLayout()`, `CONFIG`, `hexToRgba`, `seqColorscale`, `inkOn`, `errorBars`, plus `outcomeColor(outcome)` returning the series token per the Global Constraints), `lazy.ts` (`whenVisible`), `stats-fill.ts` (`fillStats`), `main.ts` registering the `CHARTS` map with every `data-chart` name used in `index.html` (`outcome-by-group`, `dropout-by-group`, `approval-band`, `effect-sizes`, `distributions`, `metrics-binary`, `metrics-multiclass`, `calibration`, `forest`, `importance`, `fairness`, `estimator`) as dynamic imports of `./charts/<camelCase>.ts` (those files are written by Tasks 8 to 10; until they exist, create each as a stub exporting `render` that calls `showError(container, "Not built yet", ...)` so the build passes).

Copy the CardioLens files named above as the starting point (read them from the reference project), then apply the identity in `docs/DESIGN.md`:
- `tokens.css`: light `--surface #f3f5f8`, `--surface-2 #e6eaf0`, `--ink #0f1a2b`, `--ink-2 #45506a`, `--ink-3 #7c869b`, `--grid #d5dbe5`, `--accent #1d4ea3`, `--series-1 #2a78d6`, `--series-2 #eb6834`, `--series-3 #1baf7a`, `--series-4 #eda100`, `--seq-1..5` `#cde2fb #9ec5f4 #5598e7 #256abf #104281`; dark `--surface #0f1622`, `--surface-2 #18222f`, `--ink #e9edf3`, `--ink-2 #b4bccb`, `--ink-3 #7f8a9d`, `--grid #26303f`, `--accent #6ea3f0`, `--series-1 #3987e5`, `--series-2 #d95926`, `--series-3 #199e70`, `--series-4 #c98500`, seq reversed as CardioLens does; both dark scopes (`prefers-color-scheme` guarded by `:not([data-theme="light"])` and `[data-theme="dark"]`). `--font-prose: "Literata Variable", Georgia, serif`; `--font-ui: "Bricolage Grotesque Variable", system-ui, sans-serif`. Import `@fontsource-variable/literata/opsz.css` and `@fontsource-variable/bricolage-grotesque/wdth.css` in `main.ts`.
- `layout.css`: at `min-width: 960px` a grid `250px minmax(0, 1fr)`; the rail (`<aside class="rail">`) is `position: sticky; top: 0; height: 100vh` holding the site name, the section list (`<nav>`), the outcome strip (`<div id="outcome-strip">`, a vertical stacked bar built by `hero.ts`), and the theme toggle; under 960px the rail is a top bar with the list scrolling horizontally and the strip as a thin horizontal bar. Prose `max-width: 66ch`; figures `max-width: 860px`; everything left aligned; works at 400px (no horizontal page scroll; plots scroll inside `.plot-scroll`).
- `hero.ts`: `renderHero()` fetches `outcome_overall.json`, draws the 10 by 10 tile grid as inline SVG into `#tiles` (tiles filled by outcome in order Dropout, Enrolled, Graduate, row-major; each tile a rounded rect with `fill` from `outcomeColor`; a 2px surface gap; `role="img"` with an `aria-label` sentence built from the counts; a `<details>` table of the three rows), fills the strip in the rail, and animates the tiles in once on load via CSS transitions unless `prefers-reduced-motion`.
- `index.html`: `<title>Dropout Lens: who leaves one Portuguese polytechnic, and what predicts it</title>`; theme boot script as CardioLens (`localStorage` key `dropout-lens-theme`); skip link; the rail; `<main>` with sections in this order and these ids and headings: `#who` "Who is in the data, and how does it end?" (hero: headline sentence "Of every 100 students who enrolled, <span data-stat="tiles_dropout">…</span> had left by the end of their course." then the tiles, then two paragraphs: the source with the citation, and a paragraph on the two course projects drawn from `docs/SOURCES.md`, and one paragraph on what "Enrolled" means); `#enrolment` "What differs at enrolment?" (figures `outcome-by-group`, `dropout-by-group`, then `approval-band` introduced as the first thing that is not an enrolment fact); `#tests` "What do the tests say?" (figures `effect-sizes`, `distributions`; prose on Holm and on effect size versus significance); `#predict` "What can a model predict, and when?" (figures `metrics-binary`, `metrics-multiclass`, `calibration`; prose on the split, CV, the two feature sets and why the second leaks); `#drivers` "What drives the model?" (figures `forest`, `importance`); `#unequal` "Where is it unequal?" (figure `fairness`; the required sentence verbatim in the prose); `#estimator` "Put a profile through the enrolment model" (figure `estimator`); `#limits` "What this cannot tell you" (a list: one institution; the outcome is measured at the normal course end so Enrolled is ambiguous and later cohorts carry more of it; first-semester features leak; tuition and debtor status are administrative states recorded during the course; no causal claims; intervals are not significance; subgroups under 30 are suppressed; parents' background and nationality are unused; the two admission-grade columns of the UCI release are absent from this copy). Footer: author, source with licence, stack, MIT, link to the repo. Every number in the prose is a `data-stat` span with a key from the `claims.json` contract; every figure is `<figure class="figure" data-chart="…"><div class="fig-skeleton"></div></figure>` with reserved height, and the three after-first-semester ones (`approval-band` and, later, toggled views) carry class `leaks` where the whole figure leaks.
- README badge target: `https://github.com/sahajm99/dropout-lens/actions/workflows/ci.yml/badge.svg`.

- [ ] Implement; `npm run typecheck` and `npm run build` pass; `npm run dev` on a free port and screenshot with `~/.claude/skills/gstack/browse/dist/browse` at 1280 and 400 px in light and dark (set `BROWSE_SERVER_PORT` to a private port) to `.superpowers/sdd/shots/`; commit `feat(site): chrome, identity, hero tiles and copy`.

## Milestone 3: figures (Tasks 8, 9, 10 in parallel; each touches only its own files)

Every chart module: `export async function render(container: HTMLElement): Promise<void>`; loads its JSON with `loadJson`; on failure `showError(container, message, () => void render(container))`; `mountFigure(container, spec)` with a title that is a sentence built from the data (template, never a typed numeral), `subtitle` with n and the interval method, `note` with the source and any suppression, `alt`, and a `table` of the plotted rows; Plotly via `import Plotly from "../plotly.ts"` with `baseLayout()` and `CONFIG`; colours only through `cssVar`/`outcomeColor`; `errorBars` for intervals; direct labels where the light-mode contrast warning applies (orange and aqua marks get value labels); re-render on theme change by listening for the `themechange` event `theme.ts` dispatches on `document`. Read CardioLens `site/src/charts/factors.ts`, `forest.ts`, `points.ts` and `heatmap.ts` for the picker, toggle and forest patterns before starting.

### Task 8: Descriptive and test figures
**Files:** `site/src/charts/outcomeByGroup.ts`, `dropoutByGroup.ts`, `approvalBand.ts`, `effectSizes.ts`, `distributions.ts`.
- `outcomeByGroup`: horizontal 100% stacked bars, one bar per level, segments Dropout/Enrolled/Graduate with 2px gaps and percentage labels inside segments over 8%; a `<select>` picker over the nine enrolment groups (skip `approval_band_s1`); suppressed rows drawn as a grey hatched bar labelled "suppressed (n under 30)"; title template "Among <lowerLead(group label)>, dropout runs from X% (<level>) to Y% (<level>)".
- `dropoutByGroup`: dot plot of the Dropout share per level with Wilson error bars, same picker (its own select), `small_n` rows greyed; title "<upperLead(level with the highest dropout share)> students left at X%, against Y% for <lowest level>".
- `approvalBand`: the `approval_band_s1` group as stacked bars; the figure has class `leaks`; subtitle contains "leaks: this is a first-semester result, not an enrolment fact"; title "Students who passed nothing in the first semester left at X%; those who passed everything, at Y%".
- `effectSizes`: horizontal bars of `cramers_v_corrected` for the 14 categorical tests sorted descending, the post-enrolment bar in `--series-2` with the `leaks` treatment and the rest in `--series-1`; hover shows raw and Holm p; a marker for the eta squared of the three numeric tests as a second small panel below (same module, Plotly subplot rows 2); title "<top label> carries the largest effect (V = X); <n_holm_sig> of <family_size> tests survive Holm adjustment".
- `distributions`: three small multiples (age, first-semester grade, second-semester grade) as quartile boxes per outcome (box from q1 to q3, median line, whiskers to min/max) built from `by_outcome`, coloured by outcome; the grade panels carry the leak marking in their subplot titles; title "Students who left were older at enrolment (median X against Y for graduates)".
- [ ] Implement; `npm run typecheck`; check each figure in the dev server at 1280 and 400 px, light and dark, screenshots to `.superpowers/sdd/shots/`; commit `feat(site): descriptive and test figures`.

### Task 9: Model figures
**Files:** `site/src/charts/metricsBinary.ts`, `metricsMulticlass.ts`, `calibration.ts`, `forest.ts`, `importance.ts`, `fairness.ts`.
- `metricsBinary`: dot plot with bootstrap error bars; y = model (four), two series = feature sets (enrolment in `--series-1`, after_s1 in `--series-2` with the leak treatment in the legend name "After first semester (leaks)"); a `<select>` for the metric (accuracy default, balanced accuracy, macro F1, ROC AUC, PR AUC); title "At enrolment the best model reaches X% accuracy (<best label>); after the first semester, Y%"; note names the split sizes and the baseline.
- `metricsMulticlass`: same layout for the three-class results; title with macro F1.
- `calibration`: reliability diagram, six lines (three models × two feature sets, the leaking ones dashed), diagonal reference in `--grid`, ≥ 8px markers, Brier in the hover; title "The enrolment models are <well|poorly> calibrated: predicted and observed dropout agree within X points on average" (compute the mean absolute gap across points for the best enrolment model).
- `forest`: log-x dot plot of odds ratios with intervals, one row per term grouped by variable, reference rows hollow at 1, `unstable` rows greyed with a note; a two-button toggle between `binary_enrolment` and `binary_after_s1` (the second applies the `leaks` class to the figure while active); title "Holding the rest constant, <strongest non-unstable term label> multiplies the odds of leaving by X".
- `importance`: horizontal bars of permutation importance (mean, sd whiskers) for the best model, a picker over the four (target, feature set) results, leak treatment when after_s1; title "<top variable> matters most to the <best label> model at enrolment".
- `fairness`: two panels (false negative rate, false positive rate), dots with Wilson error bars per level across the four groups, feature-set toggle; suppressed levels shown as a labelled gap; title "The enrolment model misses X% of the students who left among <level with the highest FNR>, against Y% among <lowest>"; the note carries the required sentence.
- [ ] Implement; typecheck; dev-server screenshots; commit `feat(site): model, driver and fairness figures`.

### Task 10: Estimator panel and parity test
**Files:** `site/src/estimator-core.ts`, `site/src/estimator-core.test.ts`, `site/src/charts/estimator.ts`.
- `estimator-core.ts`: `export interface EstimatorFile {...}` (or import from `types.ts`), `export function evaluate(est: EstimatorFile, chosen: Record<string, string>): { p: number; lo: number; hi: number }` implementing exactly the contract formula (x vector from `term_index`, `logit`, `se = sqrt(xᵀ cov x)`, sigmoid), throwing on a missing level.
- `estimator-core.test.ts` (vitest, node env): reads `../public/data/estimator.json` and `../public/data/estimator_parity.json` with `node:fs` relative to `import.meta.url`, evaluates every row, asserts `|p - row.p| < 1e-6` and the same for `lo` and `hi`; a second test asserts that all-defaults gives `sigmoid(intercept)`.
- `charts/estimator.ts`: the CardioLens panel adapted: heading "Put a profile through the enrolment model"; the eight selects from `inputs`; output sentence "Students with this profile left at X% (95% interval Y% to Z%). The training-set average was B%."; the scale with marks for this profile and the average; the held-at-reference list rendered as a sentence from `held_at_reference` ("Held at their reference level: tuition fees up to date, not a debtor, Portuguese, no special needs."); the caveat paragraph containing verbatim "an association in one Portuguese institution's 2008 to 2019 records, not a prediction about a person and not a decision tool"; never the words "risk score".
- [ ] `npm test` passes; typecheck; screenshots; commit `feat(site): browser-side estimator with interval and parity test`.

## Milestone 4: ship

### Task 11: README, PROGRESS, first push, Pages, CI green
Controller work plus one implementer for the README. README: pitch, live URL, badge, the measured results table (both feature sets, each model, accuracy and ROC AUC or macro F1 with intervals, baseline row), the strongest effect sizes, run instructions (`uv sync`, `uv run python -m pipeline`, `uv run pytest`, `cd site && npm ci && npm run dev`), checks (staleness, tests, parity), licence and citation, layout. Every number in the README is copied from `claims.json`/`model_metrics.json` at writing time and the README says which pipeline run it reflects. Commit `docs: README`.

### Task 12: QA pass on the live URL (controller)
Headless browse of the live URL at 1280 and 400 px, light and dark; console errors; every figure rendered; estimator responds; `grep` the built `index.html` for hand-typed numerals and for "risk score". Fix wave by one implementer if needed.

### Task 13: Portfolio entry
In `C:/Users/sahaj/OneDrive/Desktop/Experiments/projects/active/portfolio/portfolio-next`: copy `src/data/projects.ts` aside, `git checkout HEAD -- src/data/projects.ts`, edit the `student-retention-predictor` entry in place (title "Dropout Lens", tagline, description that corrects the record count, `category: "data-visualization"`, tags, `github`, `liveUrl`, `status: "live"`, `keyMetrics` with the true row count and the measured metrics for both feature sets, `architecture`, `techDecisions`, honest `aiContribution`, `whatIBuilt`, `whatAIHelped`), commit as sahajm99 with no trailer, push, restore the working copy and re-apply the edit, then poll the Vercel API for the deployment until READY.
