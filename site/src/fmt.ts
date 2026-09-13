/** Number formatting. Every number shown on the page passes through here. */

const nf = new Intl.NumberFormat("en-US");

/** 0.321202 -> "32.1%" */
export function pct(p: number, digits = 1): string {
  return `${(p * 100).toFixed(digits)}%`;
}

/** 0.321202 -> "32%" */
export function pct0(p: number): string {
  return pct(p, 0);
}

/** 2.015 -> "2.0x"; 31.41 -> "31x" (a second decimal on a big ratio is noise). */
export function ratio(r: number): string {
  return r >= 10 ? `${Math.round(r)}x` : `${r.toFixed(1)}x`;
}

/** 4424 -> "4,424" */
export function int(n: number): string {
  return nf.format(Math.round(n));
}

/** 0.3075, 0.3349 -> "95% CI 30.8% to 33.5%" */
export function ciText(lo: number, hi: number, digits = 1): string {
  return `95% CI ${pct(lo, digits)} to ${pct(hi, digits)}`;
}

/** 3 -> "1 in 3" */
export function oneIn(n: number): string {
  return `1 in ${int(n)}`;
}

/** 0.2345, 2 -> "0.23"; a plain fixed-point number for V, eta and odds. */
export function num(x: number, digits = 2): string {
  return x.toFixed(digits);
}

/** A p-value as the page prints it: three decimals, or "< 0.001" below that. */
export function pval(p: number): string {
  return p < 0.001 ? "< 0.001" : p.toFixed(3);
}

/** First words that keep their capital when a label starts a phrase. */
const PROPER = new Set(["Portuguese", "Holm", "Wilson", "Nursing"]);

/**
 * A label as it reads mid-sentence: the first letter drops to lower case,
 * unless the first word is an acronym ("ROC AUC"), a proper noun, or the
 * label is a multi-word name whose later words are capitalised ("Social
 * Service (evening)"), which marks it as a course name.
 */
export function lowerLead(label: string): string {
  const words = label.split(" ");
  const first = words[0] ?? "";
  if (PROPER.has(first)) return label;
  if (/[A-Z]/.test(first.slice(1))) return label;
  if (words.slice(1).some((w) => /^[A-Z]/.test(w))) return label;
  return label.charAt(0).toLowerCase() + label.slice(1);
}

/** A label that starts a sentence: the first letter rises to upper case. */
export function upperLead(label: string): string {
  return label.charAt(0).toUpperCase() + label.slice(1);
}
