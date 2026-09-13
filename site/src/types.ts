/**
 * Shapes of the JSON files written by `python -m pipeline`: the seam between
 * Python and TypeScript. Every file carries `schema`, `id` and `generated_at`.
 * Level lists are always in the categorical order `pipeline/constants.py`
 * defines, and the outcome order is Dropout, Enrolled, Graduate everywhere.
 */

export type Outcome = "Dropout" | "Enrolled" | "Graduate";

export interface CiSpec {
  method: string;
  level: number;
}

/** A proportion with its Wilson interval, `0 <= lo <= p <= hi <= 1`. */
export interface Share {
  p: number;
  lo: number;
  hi: number;
}

/** One outcome's count and share of the whole file. Never suppressed. */
export interface OutcomeRow extends Share {
  outcome: Outcome;
  n: number;
}

/** `outcome_overall.v1`: the three-way split behind the hero tiles. */
export interface OutcomeOverall {
  schema: "outcome_overall.v1";
  id: string;
  generated_at: string;
  n_total: number;
  outcomes: Outcome[];
  ci: CiSpec;
  rows: OutcomeRow[];
  /** Largest-remainder integers that sum to exactly 100. */
  tiles: Record<Outcome, number>;
}

/** One level of a group. `shares` is null when the cell is suppressed. */
export interface GroupRow {
  level: string;
  n: number;
  counts: Record<Outcome, number>;
  shares: Record<Outcome, Share> | null;
  suppressed: boolean;
  small_n: boolean;
}

export interface Group {
  id: string;
  label: string;
  /** True for the first-semester approval band, which leaks. */
  post_enrolment: boolean;
  levels: string[];
  rows: GroupRow[];
}

/** `outcome_by_group.v1`: outcome shares per level of ten groups. */
export interface OutcomeByGroup {
  schema: "outcome_by_group.v1";
  id: string;
  generated_at: string;
  outcomes: Outcome[];
  ci: CiSpec;
  suppress_below: number;
  small_n_below: number;
  n_total: number;
  groups: Group[];
}

/** A chi-square test of one categorical predictor against the outcome. */
export interface CategoricalTest {
  id: string;
  label: string;
  post_enrolment: boolean;
  test: "chi_square";
  n: number;
  levels: number;
  df: number;
  statistic: number;
  p_raw: number;
  p_holm: number;
  cramers_v: number;
  cramers_v_corrected: number;
  min_expected: number;
}

export interface AnovaResult {
  statistic: number;
  df_between: number;
  df_within: number;
  p_raw: number;
  p_holm: number;
  eta_squared: number;
}

export interface KruskalResult {
  statistic: number;
  df: number;
  p_raw: number;
  p_holm: number;
  epsilon_squared: number;
}

/** One outcome's summary of a numeric variable, with a mean interval. */
export interface OutcomeSummary {
  outcome: Outcome;
  n: number;
  mean: number;
  mean_lo: number;
  mean_hi: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
}

export interface NumericTest {
  id: string;
  label: string;
  post_enrolment: boolean;
  n: number;
  /** Rows dropped before testing (a grade of 0 means nothing was evaluated). */
  n_excluded: number;
  anova: AnovaResult;
  kruskal: KruskalResult;
  by_outcome: OutcomeSummary[];
}

/** `hypothesis_tests.v1`: the whole family, Holm-adjusted together. */
export interface HypothesisTests {
  schema: "hypothesis_tests.v1";
  id: string;
  generated_at: string;
  family_size: number;
  adjustment: "holm";
  alpha: number;
  categorical: CategoricalTest[];
  numeric: NumericTest[];
}

export type Target = "binary" | "multiclass";
export type FeatureSetId = "enrolment" | "after_s1";
export type ModelId = "majority" | "logistic" | "hgb" | "rf";

export interface FeatureSet {
  label: string;
  leaks: boolean;
  variables: string[];
}

/** A test metric with its percentile bootstrap interval. */
export interface MetricValue {
  value: number;
  lo: number;
  hi: number;
}

/** `roc_auc` and `pr_auc` are null for the majority baseline. */
export interface TestMetrics {
  accuracy: MetricValue;
  balanced_accuracy: MetricValue;
  macro_f1: MetricValue;
  roc_auc: MetricValue | null;
  pr_auc: MetricValue | null;
}

export interface ModelResult {
  target: Target;
  feature_set: FeatureSetId;
  model: ModelId;
  leaks: boolean;
  /** Null for the majority baseline and for logistic regression. */
  chosen_params: Record<string, number | string | null> | null;
  cv: { macro_f1_mean: number; macro_f1_sd: number };
  test: TestMetrics;
  confusion: number[][];
  classes: string[];
}

/** `model_metrics.v1`: 16 results, 2 targets x 2 feature sets x 4 models. */
export interface ModelMetrics {
  schema: "model_metrics.v1";
  id: string;
  generated_at: string;
  seed: number;
  split: {
    n_train: number;
    n_test: number;
    test_share: number;
    stratified_on: string;
  };
  cv: { folds: number; selection_metric: string };
  bootstrap: { resamples: number; level: number };
  feature_sets: Record<FeatureSetId, FeatureSet>;
  targets: {
    binary: {
      label: string;
      classes: string[];
      positive: string;
      test_positive_share: number;
    };
    multiclass: { label: string; classes: string[] };
  };
  models: ModelId[];
  model_labels: Record<ModelId, string>;
  results: ModelResult[];
  /** The model with the highest CV macro F1 per target and feature set. */
  best: Record<Target, Record<FeatureSetId, ModelId>>;
}

export interface CalibrationPoint {
  mean_predicted: number;
  fraction_positive: number;
  n: number;
}

export interface CalibrationCurve {
  feature_set: FeatureSetId;
  model: ModelId;
  leaks: boolean;
  brier: number;
  points: CalibrationPoint[];
}

/** `calibration.v1`: six reliability curves for the binary models. */
export interface Calibration {
  schema: "calibration.v1";
  id: string;
  generated_at: string;
  bins: number;
  strategy: string;
  target: "binary";
  curves: CalibrationCurve[];
}

/**
 * One coefficient of a logistic model, as an odds ratio. A reference level
 * carries `or: 1` with null `lo`, `hi`, `se` and `p_value`; `unstable` is true
 * when any cell of the level-versus-outcome table is under 10.
 */
export interface OddsTerm {
  variable: string;
  level: string;
  reference: string;
  label: string;
  or: number;
  lo: number | null;
  hi: number | null;
  coef: number;
  se: number | null;
  p_value: number | null;
  n_level: number;
  events_level: number;
  is_reference: boolean;
  unstable: boolean;
}

export interface BinaryOddsModel {
  label: string;
  target: "binary";
  feature_set: FeatureSetId;
  leaks: boolean;
  covariates: string[];
  n_obs: number;
  n_events: number;
  converged: boolean;
  pseudo_r2_mcfadden: number;
  intercept: number;
  terms: OddsTerm[];
}

/** The CardioLens name for a binary model, kept for copied chart code. */
export type OddsModel = BinaryOddsModel;

export interface OddsContrast {
  intercept: number;
  terms: OddsTerm[];
}

export interface MulticlassOddsModel {
  label: string;
  target: "multiclass";
  feature_set: FeatureSetId;
  leaks: boolean;
  base: Outcome;
  covariates: string[];
  n_obs: number;
  converged: boolean;
  pseudo_r2_mcfadden: number;
  contrasts: Record<"Dropout" | "Enrolled", OddsContrast>;
}

/** `odds_ratios.v1`: the pinned-reference logistic models, fitted on train. */
export interface OddsRatios {
  schema: "odds_ratios.v1";
  id: string;
  generated_at: string;
  fitted_on: string;
  models: {
    binary_enrolment: BinaryOddsModel;
    binary_after_s1: BinaryOddsModel;
    multiclass_enrolment: MulticlassOddsModel;
  };
}

export interface ImportanceFeature {
  variable: string;
  label: string;
  mean: number;
  sd: number;
}

export interface ImportanceResult {
  target: Target;
  feature_set: FeatureSetId;
  model: ModelId;
  leaks: boolean;
  baseline_score: number;
  /** Sorted by `mean` descending; a variable is permuted as a whole. */
  features: ImportanceFeature[];
}

/** `permutation_importance.v1`: the best model per target and feature set. */
export interface PermutationImportance {
  schema: "permutation_importance.v1";
  id: string;
  generated_at: string;
  repeats: number;
  seed: number;
  scoring: Record<Target, string>;
  results: ImportanceResult[];
}

/** `fnr` or `fpr` is null, with its `suppressed_*` flag true, under 30 rows. */
export interface FairnessLevel {
  level: string;
  n: number;
  n_positive: number;
  n_negative: number;
  fnr: Share | null;
  fpr: Share | null;
  suppressed_fnr: boolean;
  suppressed_fpr: boolean;
}

export interface FairnessGroup {
  id: string;
  label: string;
  levels: FairnessLevel[];
}

export interface FairnessResult {
  feature_set: FeatureSetId;
  model: ModelId;
  leaks: boolean;
  n_test: number;
  groups: FairnessGroup[];
}

/** `fairness.v1`: error rates per group of the best binary model. */
export interface Fairness {
  schema: "fairness.v1";
  id: string;
  generated_at: string;
  target: "binary";
  threshold: number;
  ci: CiSpec;
  suppress_below: number;
  results: FairnessResult[];
}

/** One question of the estimator form. The form is built only from these. */
export interface EstimatorInput {
  variable: string;
  label: string;
  levels: string[];
  default: string;
}

export interface HeldAtReference {
  variable: string;
  label: string;
  level: string;
}

/**
 * `estimator.v1`: the enrolment binary Logit on the logit scale, with its
 * coefficient covariance, enough to evaluate one profile with a delta-method
 * interval in the browser. `terms[0]` is "Intercept"; `term_index[variable]
 * [level]` is the position of a non-reference level's term in `terms` and
 * in `cov`; reference levels have no entry.
 */
export interface Estimator {
  schema: "estimator.v1";
  id: string;
  generated_at: string;
  model: string;
  fitted_on: string;
  z: number;
  intercept: number;
  base_rate: number;
  n_obs: number;
  inputs: EstimatorInput[];
  held_at_reference: HeldAtReference[];
  coefficients: Record<string, Record<string, number>>;
  terms: string[];
  term_index: Record<string, Record<string, number>>;
  cov: number[][];
}

/** The name the estimator core uses for the same file. */
export type EstimatorFile = Estimator;

export interface ParityRow {
  inputs: Record<string, string>;
  p: number;
  lo: number;
  hi: number;
}

/** `estimator_parity.v1`: 100 profiles the browser must reproduce to 1e-6. */
export interface EstimatorParity {
  schema: "estimator_parity.v1";
  id: string;
  generated_at: string;
  seed: number;
  rows: ParityRow[];
}

/** `claims.v1`: a flat bag of named numbers and labels quoted in the prose. */
export type Claims = Record<string, number | string>;

/** `meta.v1`: provenance of the source file and the generated outputs. */
export interface Meta {
  schema: "meta.v1";
  id: string;
  generated_at: string;
  source_file: string;
  sha256: string;
  n_rows: number;
  n_columns: number;
  n_features: number;
  n_duplicate_rows: number;
  outcome_counts: Record<Outcome, number>;
  seed: number;
  files: string[];
}
