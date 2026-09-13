import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { int, num } from "../fmt.ts";
import Plotly from "../plotly.ts";
import type {
  HypothesisTests,
  NumericTest,
  Outcome,
  OutcomeSummary,
} from "../types.ts";
import {
  CONFIG,
  OUTCOMES,
  cssVar,
  hexToRgba,
  horizontalLegend,
  layoutTemplate,
  outcomeColor,
} from "./theme.ts";

const VARIABLE_IDS = ["age", "grade_s1", "grade_s2"] as const;

function byId(data: HypothesisTests, id: string): NumericTest | undefined {
  return data.numeric.find((t) => t.id === id);
}

function summaryFor(
  test: NumericTest | undefined,
  outcome: Outcome,
): OutcomeSummary | undefined {
  return test?.by_outcome.find((r) => r.outcome === outcome);
}

function panelTitle(test: NumericTest): string {
  return test.post_enrolment ? `${test.label} (leaks)` : test.label;
}

function specFor(data: HypothesisTests): FigureSpec {
  const age = byId(data, "age");
  const dropoutAge = summaryFor(age, "Dropout");
  const graduateAge = summaryFor(age, "Graduate");
  const title =
    age && dropoutAge && graduateAge
      ? `Students who left were older at enrolment (median ${int(dropoutAge.median)} against ${int(graduateAge.median)} for graduates)`
      : "Age and grade distributions by outcome";

  const variables = VARIABLE_IDS.map((id) => byId(data, id)).filter(
    (t): t is NumericTest => t !== undefined,
  );

  return {
    id: "fig-distributions",
    title,
    subtitle:
      "Quartile boxes (Q1 to Q3) with a median line and whiskers to min and max, by outcome",
    note: "Source: Realinho, Machado, Baptista and Martins, UCI Machine Learning Repository. The first- and second-semester grade panels leak: those columns are recorded at the end of the semester, not on enrolment day. Rows with no recorded units are excluded from the grade panels.",
    alt: `Three small-multiple box plots by outcome: age at enrolment, first-semester grade and second-semester grade. The grade panels leak. ${title}.`,
    table: {
      columns: [
        "Variable",
        "Outcome",
        "Students",
        "Min",
        "Q1",
        "Median",
        "Q3",
        "Max",
      ],
      rows: variables.flatMap((test) =>
        OUTCOMES.map((outcome) => {
          const row = summaryFor(test, outcome);
          return [
            panelTitle(test),
            outcome,
            row ? int(row.n) : "—",
            row ? num(row.min) : "—",
            row ? num(row.q1) : "—",
            row ? num(row.median) : "—",
            row ? num(row.q3) : "—",
            row ? num(row.max) : "—",
          ];
        }),
      ),
    },
  };
}

/** The box trace's precomputed-stats fields, which the shared Plotly typings omit. */
interface BoxStatsTrace extends Partial<Plotly.PlotData> {
  q1: number[];
  median: number[];
  q3: number[];
  lowerfence: number[];
  upperfence: number[];
}

const AXES: [Plotly.XAxisName, Plotly.YAxisName][] = [
  ["x", "y"],
  ["x2", "y2"],
  ["x3", "y3"],
];

function boxTrace(
  outcome: Outcome,
  row: OutcomeSummary,
  color: string,
  xref: Plotly.XAxisName,
  yref: Plotly.YAxisName,
  showlegend: boolean,
): BoxStatsTrace {
  return {
    type: "box",
    xaxis: xref,
    yaxis: yref,
    name: outcome,
    legendgroup: outcome,
    showlegend,
    x: [outcome],
    q1: [row.q1],
    median: [row.median],
    q3: [row.q3],
    lowerfence: [row.min],
    upperfence: [row.max],
    marker: { color },
    line: { color },
    fillcolor: hexToRgba(color, 0.35),
    boxpoints: false,
  };
}

function heading(text: string, xCenter: number, ink: string): Partial<Plotly.Annotations> {
  return {
    text,
    xref: "paper",
    yref: "paper",
    x: xCenter,
    y: 1.04,
    xanchor: "center",
    yanchor: "bottom",
    showarrow: false,
    font: { color: ink, size: 12 },
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

  const variables = VARIABLE_IDS.map((id) => byId(data, id)).filter(
    (t): t is NumericTest => t !== undefined,
  );
  if (variables.length !== VARIABLE_IDS.length) {
    showError(
      container,
      "The test file is missing one of the three distributions this chart draws.",
      () => void render(container),
    );
    return;
  }

  const plot = mountFigure(container, specFor(data));

  function draw(): Promise<unknown> {
    const ink = cssVar("--ink-2");
    const gap = 0.06;
    const width = (1 - 2 * gap) / 3;
    const domains: [number, number][] = [0, 1, 2].map((i) => [
      i * (width + gap),
      i * (width + gap) + width,
    ]);

    const traces: BoxStatsTrace[] = [];
    variables.forEach((test, i) => {
      const axes = AXES[i]!;
      OUTCOMES.forEach((outcome) => {
        const row = summaryFor(test, outcome);
        if (!row) return;
        traces.push(
          boxTrace(outcome, row, outcomeColor(outcome), axes[0], axes[1], i === 0),
        );
      });
    });

    const base = layoutTemplate();
    const xAxis = {
      ...base.xaxis,
      type: "category" as const,
      showgrid: false,
    };
    const yAxis = {
      ...base.yaxis,
      automargin: true,
      domain: [0, 1] as [number, number],
    };
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: true,
      legend: horizontalLegend(),
      margin: { ...base.margin, t: 40 },
      grid: { rows: 1, columns: 3, pattern: "independent" },
      xaxis: { ...xAxis, anchor: "y", domain: domains[0] },
      xaxis2: { ...xAxis, anchor: "y2", domain: domains[1] },
      xaxis3: { ...xAxis, anchor: "y3", domain: domains[2] },
      yaxis: { ...yAxis, anchor: "x" },
      yaxis2: { ...yAxis, anchor: "x2" },
      yaxis3: { ...yAxis, anchor: "x3" },
      annotations: variables.map((test, i) =>
        heading(
          panelTitle(test),
          (domains[i]![0] + domains[i]![1]) / 2,
          ink,
        ),
      ),
    };
    return Plotly.react(
      plot,
      traces as unknown as Partial<Plotly.PlotData>[],
      layout,
      CONFIG,
    );
  }

  document.addEventListener("themechange", () => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
