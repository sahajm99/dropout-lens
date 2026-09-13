/**
 * The browser twin of `pipeline/estimator.py::evaluate`.
 *
 * `estimator.json` carries the enrolment binary Logit's coefficients and
 * their covariance, addressed by `term_index`, so a profile chosen in the
 * browser can be turned into a probability with a delta-method interval
 * without re-fitting anything. This file must stay numerically identical to
 * the Python function it mirrors; `estimator-core.test.ts` checks that
 * against every row of `estimator_parity.json`.
 */

export type { EstimatorFile, EstimatorInput, HeldAtReference } from "./types.ts";

import type { EstimatorFile } from "./types.ts";

export interface EstimatorResult {
  p: number;
  lo: number;
  hi: number;
}

/** Logistic function in the overflow-safe form, matching `pipeline/estimator.py::sigmoid`. */
function sigmoid(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

/**
 * `(p, lo, hi)` for one profile: builds the design vector `x` from
 * `term_index` (1 at index 0 for the intercept, and at each chosen
 * non-reference level's term index), computes `logit = x . beta`,
 * `se = sqrt(x^T cov x)`, and transforms with `sigmoid`.
 *
 * `chosen` must name exactly `est.inputs`' variables. A variable this
 * function does not recognise, or a level outside that input's list, throws
 * rather than silently falling back to the reference level - the held-at-
 * reference covariates are not in `chosen` and contribute nothing, exactly
 * as in the Python twin.
 */
export function evaluate(
  est: EstimatorFile,
  chosen: Record<string, string>,
): EstimatorResult {
  const inputNames = est.inputs.map((i) => i.variable);
  const chosenNames = Object.keys(chosen);
  const wantedSet = new Set(inputNames);
  const gotSet = new Set(chosenNames);
  const sameSet =
    wantedSet.size === gotSet.size &&
    [...wantedSet].every((v) => gotSet.has(v));
  if (!sameSet) {
    throw new Error(
      `chosen must name exactly ${inputNames.join(", ")}; got ${chosenNames.join(", ")}.`,
    );
  }

  const k = est.terms.length;
  const beta = new Array<number>(k).fill(0);
  beta[0] = est.intercept;
  for (const [variable, levels] of Object.entries(est.term_index)) {
    for (const [level, index] of Object.entries(levels)) {
      const coef = est.coefficients[variable]?.[level];
      if (typeof coef !== "number" || !Number.isFinite(coef)) {
        throw new Error(`No coefficient for ${variable} = "${level}".`);
      }
      beta[index] = coef;
    }
  }

  const x = new Array<number>(k).fill(0);
  x[0] = 1;
  for (const input of est.inputs) {
    const level = chosen[input.variable];
    if (level === undefined) {
      throw new Error(`No level chosen for "${input.variable}".`);
    }
    if (!input.levels.includes(level)) {
      throw new Error(
        `${input.variable}: "${level}" is not one of ${input.levels.join(", ")}.`,
      );
    }
    const index = est.term_index[input.variable]?.[level];
    if (index !== undefined) x[index] = 1;
  }

  const active: number[] = [];
  for (let i = 0; i < k; i++) if (x[i] !== 0) active.push(i);

  let logit = 0;
  for (const i of active) logit += x[i] * beta[i];

  let variance = 0;
  for (const i of active) {
    const row = est.cov[i];
    if (!row) throw new Error(`Covariance matrix is missing row ${i}.`);
    for (const j of active) variance += x[i] * row[j] * x[j];
  }
  if (variance < 0) {
    throw new Error("Negative variance from the published covariance matrix.");
  }
  const se = Math.sqrt(variance);
  const z = est.z;

  return {
    p: sigmoid(logit),
    lo: sigmoid(logit - z * se),
    hi: sigmoid(logit + z * se),
  };
}
