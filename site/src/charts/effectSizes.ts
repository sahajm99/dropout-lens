import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { int, num, pval, upperLead } from "../fmt.ts";
import Plotly from "../plotly.ts";
import type {
  CategoricalTest,
  HypothesisTests,
  NumericTest,
} from "../types.ts";
import { CONFIG, cssVar, layoutTemplate, reversed, series } from "./theme.ts";

/** Every p-Holm in the twenty-test family under alpha: categorical, then both tests of each numeric variable. */
function countHolmSig(data: HypothesisTests): number {
  let n = 0;
  for (const c of data.categorical) if (c.p_holm < data.alpha) n += 1;
  for (const t of data.numeric) {
    if (t.anova.p_holm < data.alpha) n += 1;
    if (t.kruskal.p_holm < data.alpha) n += 1;
  }
  return n;
}

function specFor(data: HypothesisTests): FigureSpec {
  const sorted = [...data.categorical].sort(
    (a, b) => b.cramers_v_corrected - a.cramers_v_corrected,
  );
  const top = sorted[0];
  const holmSig = countHolmSig(data);
  const title = top
    ? `${upperLead(top.label)} carries the largest effect (V = ${num(top.cramers_v_corrected)}); ${int(holmSig)} of ${int(data.family_size)} tests survive Holm adjustment`
    : `${int(holmSig)} of ${int(data.family_size)} tests survive Holm adjustment`;
  return {
    id: "fig-effect-sizes",
    title,
    subtitle: `Bias-corrected Cramér's V for ${int(data.categorical.length)} categorical variables and eta squared (ANOVA) for ${int(data.numeric.length)} numeric variables; Holm-adjusted across a family of ${int(data.family_size)} at alpha = ${data.alpha}`,
    note: "Source: chi-square tests (categorical) and one-way ANOVA (numeric). Outlined bars and markers are fitted on the first-semester approval band and leak: they are not enrolment facts.",
    alt: `Two panels. The top panel is a horizontal bar chart of bias-corrected Cramér's V for ${int(data.categorical.length)} categorical variables, sorted descending. The bottom panel is a dot plot of eta squared for ${int(data.numeric.length)} numeric variables. ${title}.`,
    table: {
      columns: ["Variable", "Measure", "Effect size", "p (raw)", "p (Holm)"],
      rows: [
        ...sorted.map((t) => [
          t.label,
          "Cramér's V (corrected)",
          num(t.cramers_v_corrected),
          pval(t.p_raw),
          pval(t.p_holm),
        ]),
        ...data.numeric.map((t) => [
          t.label,
          "Eta squared (ANOVA)",
          num(t.anova.eta_squared),
          pval(t.anova.p_raw),
          pval(t.anova.p_holm),
        ]),
      ],
    },
  };
}

function heading(
  text: string,
  y: number,
  ink: string,
): Partial<Plotly.Annotations> {
  return {
    text: `<b>${text}</b>`,
    xref: "paper",
    yref: "paper",
    x: 0,
    y,
    xanchor: "left",
    yanchor: "bottom",
    showarrow: false,
    font: { color: ink, size: 12 },
  };
}

function categoricalTrace(
  tests: CategoricalTest[],
  color: string,
  leakColor: string,
  accent: string,
): Partial<Plotly.PlotData> {
  const ts = reversed(tests);
  return {
    type: "bar",
    orientation: "h",
    xaxis: "x",
    yaxis: "y",
    showlegend: false,
    y: ts.map((t) => t.label),
    x: ts.map((t) => t.cramers_v_corrected),
    customdata: ts.map((t) => [pval(t.p_raw), pval(t.p_holm)]),
    hovertemplate:
      "<b>%{y}</b><br>V = %{x:.3f}<br>p (raw) %{customdata[0]}<br>p (Holm) %{customdata[1]}<extra></extra>",
    marker: {
      color: ts.map((t) => (t.post_enrolment ? leakColor : color)),
      line: {
        color: accent,
        width: ts.map((t) => (t.post_enrolment ? 2 : 0)),
      },
    },
  };
}

function numericTrace(
  tests: NumericTest[],
  color: string,
  leakColor: string,
  accent: string,
): Partial<Plotly.PlotData> {
  const ts = reversed(tests);
  return {
    type: "scatter",
    mode: "markers",
    xaxis: "x2",
    yaxis: "y2",
    showlegend: false,
    y: ts.map((t) => t.label),
    x: ts.map((t) => t.anova.eta_squared),
    customdata: ts.map((t) => [pval(t.anova.p_raw), pval(t.anova.p_holm)]),
    hovertemplate:
      "<b>%{y}</b><br>eta squared = %{x:.3f}<br>p (raw) %{customdata[0]}<br>p (Holm) %{customdata[1]}<extra></extra>",
    marker: {
      symbol: "circle",
      size: 12,
      color: ts.map((t) => (t.post_enrolment ? leakColor : color)),
      line: {
        color: accent,
        width: ts.map((t) => (t.post_enrolment ? 2 : 0)),
      },
    },
  };
}

export async function render(container: HTMLElement): Promise<void> {
  let data: HypothesisTests;
  try {
    data = await loadJson<HypothesisTests>("hypothesis_tests.json");
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the test data.",
      () => void render(container),
    );
    return;
  }

  if (!data.categorical.length || !data.numeric.length) {
    showError(container, "The test file carries no tests to plot.", () =>
      void render(container),
    );
    return;
  }

  const sorted = [...data.categorical].sort(
    (a, b) => b.cramers_v_corrected - a.cramers_v_corrected,
  );

  const plot = mountFigure(container, specFor(data));

  function draw(): Promise<unknown> {
    const c = series();
    const ink = cssVar("--ink");
    const traces = [
      categoricalTrace(sorted, c.s1, c.s2, c.accent),
      numericTrace(data.numeric, c.s1, c.s2, c.accent),
    ];

    const gap = 0.12;
    const topN = sorted.length;
    const bottomN = data.numeric.length;
    const share = (1 - gap) * (bottomN / (topN + bottomN));
    const bottomDomain: [number, number] = [0, share];
    const topDomain: [number, number] = [share + gap, 1];

    const vMax = Math.max(...sorted.map((t) => t.cramers_v_corrected));
    const etaMax = Math.max(...data.numeric.map((t) => t.anova.eta_squared));

    const base = layoutTemplate();
    const yAxis = {
      ...base.yaxis,
      automargin: true,
      showgrid: false,
      categoryorder: "array" as const,
    };
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: false,
      hovermode: "closest",
      margin: { ...base.margin, t: 30, b: 44 },
      grid: { rows: 2, columns: 1, roworder: "top to bottom" },
      xaxis: {
        ...base.xaxis,
        anchor: "y",
        domain: [0, 1],
        range: [0, vMax * 1.15],
        title: { text: "Cramér's V (bias-corrected)", standoff: 6 },
      },
      xaxis2: {
        ...base.xaxis,
        anchor: "y2",
        domain: [0, 1],
        range: [0, etaMax * 1.15],
        title: { text: "Eta squared (ANOVA)", standoff: 6 },
      },
      yaxis: {
        ...yAxis,
        domain: topDomain,
        categoryarray: reversed(sorted.map((t) => t.label)),
      },
      yaxis2: {
        ...yAxis,
        anchor: "x2",
        domain: bottomDomain,
        categoryarray: reversed(data.numeric.map((t) => t.label)),
      },
      annotations: [
        heading("Categorical variables", 1.01, ink),
        heading("Numeric variables", bottomDomain[1] + 0.012, ink),
      ],
    };
    return Plotly.react(plot, traces, layout, CONFIG);
  }

  document.addEventListener("themechange", () => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
