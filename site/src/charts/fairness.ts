import { loadJson } from "../data.ts";
import {
  controlSlot,
  mountFigure,
  showError,
  updateFigure,
  type FigureSpec,
} from "../figure.ts";
import { ciText, int, lowerLead, pct, pct0 } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { onThemeChange } from "../theme.ts";
import type { Fairness, FairnessResult, FeatureSetId } from "../types.ts";
import { CONFIG, cssVar, errorBars, layoutTemplate, reversed, series } from "./theme.ts";

/** Required verbatim, in the figure's own note. */
const REQUIRED_NOTE =
  "A gap here is a property of this model on this data, not a finding about students.";

const FEATURE_SET_LABEL: Record<FeatureSetId, string> = {
  enrolment: "At enrolment",
  after_s1: "After first semester (leaks)",
};

interface Row {
  group: string;
  level: string;
  label: string;
  n: number;
  p: number | null;
  lo: number | null;
  hi: number | null;
}

function rowsFor(result: FairnessResult, key: "fnr" | "fpr"): Row[] {
  const out: Row[] = [];
  for (const g of result.groups) {
    for (const lvl of g.levels) {
      const share = key === "fnr" ? lvl.fnr : lvl.fpr;
      out.push({
        group: g.label,
        level: lvl.level,
        label: `${g.label}: ${lvl.level}`,
        n: key === "fnr" ? lvl.n_positive : lvl.n_negative,
        p: share?.p ?? null,
        lo: share?.lo ?? null,
        hi: share?.hi ?? null,
      });
    }
  }
  return out;
}

function extreme(rows: Row[], pick: "max" | "min"): Row | undefined {
  const defined = rows.filter((r): r is Row & { p: number } => r.p !== null);
  if (!defined.length) return undefined;
  return defined.reduce((a, b) =>
    (pick === "max" ? b.p > a.p : b.p < a.p) ? b : a,
  );
}

function specFor(data: Fairness, result: FairnessResult): FigureSpec {
  const fnrRows = rowsFor(result, "fnr");
  const worst = extreme(fnrRows, "max");
  const best = extreme(fnrRows, "min");
  const phrase = result.feature_set === "enrolment" ? "enrolment" : "after-first-semester";

  const title =
    worst && best
      ? `The ${phrase} model misses ${pct(worst.p as number)} of the students who left among ${lowerLead(worst.level)}, against ${pct(best.p as number)} among ${lowerLead(best.level)}`
      : `False negative and false positive rates by group, ${FEATURE_SET_LABEL[result.feature_set]}`;

  const rows: (string | number)[][] = [];
  for (const g of result.groups) {
    for (const lvl of g.levels) {
      rows.push([
        g.label,
        lvl.level,
        int(lvl.n),
        lvl.suppressed_fnr || !lvl.fnr ? "suppressed (n < 30)" : pct(lvl.fnr.p),
        lvl.suppressed_fnr || !lvl.fnr ? "—" : ciText(lvl.fnr.lo, lvl.fnr.hi),
        lvl.suppressed_fpr || !lvl.fpr ? "suppressed (n < 30)" : pct(lvl.fpr.p),
        lvl.suppressed_fpr || !lvl.fpr ? "—" : ciText(lvl.fpr.lo, lvl.fpr.hi),
      ]);
    }
  }

  return {
    id: "fig-fairness",
    title,
    subtitle: `${int(result.n_test)} test students, Wilson 95% intervals, ${FEATURE_SET_LABEL[result.feature_set]}`,
    note: `${REQUIRED_NOTE} Threshold ${pct0(data.threshold)} on the test predictions; a level with fewer than ${int(data.suppress_below)} students in the denominator is shown as a labelled gap rather than a rate.`,
    alt: `Two dot plots with Wilson 95% intervals: false negative rate and false positive rate by level of gender, scholarship, age band and tuition fee status, ${FEATURE_SET_LABEL[result.feature_set].toLowerCase()}. ${title}.`,
    table: {
      columns: ["Group", "Level", "n", "FNR", "FNR 95% CI", "FPR", "FPR 95% CI"],
      rows,
    },
  };
}

function panelTrace(rows: Row[], axis: "y" | "y2", color: string): Partial<Plotly.PlotData> {
  const drawable = reversed(rows.filter((r): r is Row & { p: number; lo: number; hi: number } =>
    r.p !== null && r.lo !== null && r.hi !== null,
  ));
  return {
    type: "scatter",
    mode: "markers",
    yaxis: axis,
    xaxis: "x",
    showlegend: false,
    y: drawable.map((r) => r.label),
    x: drawable.map((r) => r.p),
    customdata: drawable.map((r) => [ciText(r.lo, r.hi), r.n]),
    hovertemplate: "<b>%{x:.1%}</b> (%{customdata[0]})<br>n = %{customdata[1]:,}<extra>%{y}</extra>",
    marker: { color, size: 9 },
    error_x: errorBars(
      drawable.map((r) => r.p),
      drawable.map((r) => r.lo),
      drawable.map((r) => r.hi),
      color,
    ),
  };
}

function suppressedAnnotations(
  rows: Row[],
  axis: "y" | "y2",
  color: string,
): Partial<Plotly.Annotations>[] {
  return rows
    .filter((r) => r.p === null)
    .map((r) => ({
      x: 0.015,
      y: r.label,
      xref: "x",
      yref: axis,
      xanchor: "left",
      yanchor: "middle",
      showarrow: false,
      text: "suppressed",
      font: { color, size: 10 },
    }));
}

function toggle(
  selected: FeatureSetId,
  onChange: (key: FeatureSetId) => void,
): HTMLDivElement {
  const group = document.createElement("div");
  group.className = "seg";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Feature set");
  const order: FeatureSetId[] = ["enrolment", "after_s1"];
  for (const key of order) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "seg-btn";
    btn.textContent = FEATURE_SET_LABEL[key];
    btn.setAttribute("aria-pressed", String(key === selected));
    btn.addEventListener("click", () => {
      for (const other of group.querySelectorAll("button")) {
        other.setAttribute("aria-pressed", String(other === btn));
      }
      onChange(key);
    });
    group.appendChild(btn);
  }
  return group;
}

const heading = (text: string, y: number, ink: string): Partial<Plotly.Annotations> => ({
  text: `<b>${text}</b>`,
  xref: "paper",
  yref: "paper",
  x: 0,
  y,
  xanchor: "left",
  yanchor: "bottom",
  showarrow: false,
  font: { color: ink, size: 12 },
});

export async function render(container: HTMLElement): Promise<void> {
  let data: Fairness;
  try {
    data = await loadJson<Fairness>("fairness.json");
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the fairness data.",
      () => void render(container),
    );
    return;
  }

  const first = data.results[0];
  if (!first) {
    showError(container, "The fairness file carries no results.", () =>
      void render(container),
    );
    return;
  }
  let result: FairnessResult = first;
  container.classList.toggle("leaks", result.leaks);

  const plot = mountFigure(container, specFor(data, result));
  controlSlot(plot).before(
    toggle(result.feature_set, (key) => {
      const next = data.results.find((r) => r.feature_set === key);
      if (!next) return;
      result = next;
      container.classList.toggle("leaks", result.leaks);
      updateFigure(container, specFor(data, result));
      void draw();
    }),
  );

  // A fixed gap between the two panels, so the heading above the bottom
  // panel never collides with the top panel's lowest row.
  const gap = 0.1;
  const half = (1 - gap) / 2;
  const topDomain: [number, number] = [half + gap, 1];
  const bottomDomain: [number, number] = [0, half];

  function draw(): Promise<unknown> {
    const c = series();
    const ink = cssVar("--ink");
    const ink3 = cssVar("--ink-3");
    const fnrRows = rowsFor(result, "fnr");
    const fprRows = rowsFor(result, "fpr");
    const categoryarray = reversed(fnrRows.map((r) => r.label));

    const traces = [panelTrace(fnrRows, "y", c.s1), panelTrace(fprRows, "y2", c.s1)];
    const annotations: Partial<Plotly.Annotations>[] = [
      ...suppressedAnnotations(fnrRows, "y", ink3),
      ...suppressedAnnotations(fprRows, "y2", ink3),
      heading("False negative rate: missed the students who left", 1.005, ink),
      heading("False positive rate: flagged students who did not leave", bottomDomain[1] + 0.02, ink),
    ];

    const base = layoutTemplate();
    const yAxis = {
      ...base.yaxis,
      automargin: true,
      showgrid: false,
      anchor: "x" as const,
      categoryorder: "array" as const,
      categoryarray,
      tickfont: { color: cssVar("--ink-2"), size: 11 },
    };
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: false,
      hovermode: "closest",
      margin: { ...base.margin, t: 26, b: 40 },
      grid: { rows: 2, columns: 1, roworder: "top to bottom" },
      annotations,
      xaxis: {
        ...base.xaxis,
        anchor: "y2",
        domain: [0, 1],
        range: [0, 1],
        tickformat: ".0%",
        title: { text: "Rate", standoff: 6 },
      },
      yaxis: { ...yAxis, domain: topDomain },
      yaxis2: { ...yAxis, domain: bottomDomain },
    };
    return Plotly.react(plot, traces, layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
