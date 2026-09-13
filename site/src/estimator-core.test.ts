/// <reference types="node" />
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { evaluate } from "./estimator-core.ts";
import type { EstimatorFile, EstimatorParity } from "./types.ts";

function readJson<T>(relativePath: string): T {
  const url = new URL(relativePath, import.meta.url);
  return JSON.parse(readFileSync(url, "utf8")) as T;
}

const estimator = readJson<EstimatorFile>("../public/data/estimator.json");
const parity = readJson<EstimatorParity>("../public/data/estimator_parity.json");

describe("evaluate", () => {
  it("reproduces every row of estimator_parity.json to 1e-6", () => {
    expect(parity.rows.length).toBeGreaterThan(0);
    for (const row of parity.rows) {
      const { p, lo, hi } = evaluate(estimator, row.inputs);
      expect(Math.abs(p - row.p)).toBeLessThan(1e-6);
      expect(Math.abs(lo - row.lo)).toBeLessThan(1e-6);
      expect(Math.abs(hi - row.hi)).toBeLessThan(1e-6);
    }
  });

  it("gives sigmoid(intercept) when every input is left at its default", () => {
    const chosen: Record<string, string> = {};
    for (const input of estimator.inputs) chosen[input.variable] = input.default;

    const { p } = evaluate(estimator, chosen);
    const expected = 1 / (1 + Math.exp(-estimator.intercept));
    expect(Math.abs(p - expected)).toBeLessThan(1e-9);
  });
});
