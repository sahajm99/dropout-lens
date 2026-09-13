import {
  int,
  lowerLead,
  num,
  oneIn,
  pct,
  pct0,
  pval,
  ratio,
  upperLead,
} from "./fmt.ts";

type NumberFormatter = (value: number) => string;
type StringFormatter = (value: string) => string;

const NUMBER_FORMATTERS: Record<string, NumberFormatter> = {
  pct,
  pct0,
  ratio,
  int,
  oneIn,
  pval,
  num1: (v) => num(v, 1),
  num2: (v) => num(v, 2),
  num3: (v) => num(v, 3),
  /** A difference between two shares, in percentage points, unsigned. */
  pts: (v) => `${(Math.abs(v) * 100).toFixed(1)} points`,
  raw: (v) => String(v),
};

const STRING_FORMATTERS: Record<string, StringFormatter> = {
  lower: lowerLead,
  upper: upperLead,
  raw: (v) => v,
};

/**
 * Fill every `<span data-stat="key|fmt">` from claims.json, so no number that
 * describes the data is hand-typed in the HTML. A label (a string value) may
 * ask for `lower` or `upper` to sit mid-sentence or start one.
 */
export function fillStats(claims: Record<string, number | string>): void {
  for (const node of document.querySelectorAll<HTMLElement>("[data-stat]")) {
    const spec = node.dataset.stat ?? "";
    const [key, fmtName = "raw"] = spec.split("|");
    const value = claims[key];
    if (value === undefined) {
      console.warn(`stats-fill: claims.json has no key "${key}"`);
      continue;
    }
    if (typeof value === "string") {
      const fmt = STRING_FORMATTERS[fmtName] ?? STRING_FORMATTERS.raw;
      node.textContent = fmt(value);
      continue;
    }
    const fmt = NUMBER_FORMATTERS[fmtName];
    if (!fmt) {
      console.warn(`stats-fill: unknown format "${fmtName}" for "${key}"`);
      continue;
    }
    node.textContent = fmt(value);
  }
}
