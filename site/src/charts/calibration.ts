import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { int, num, pct } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { onThemeChange } from "../theme.ts";
import type { Calibration, CalibrationCurve, FeatureSetId, ModelId } from "../types.ts";
import { CONFIG, cssVar, horizontalLegend, layoutTemplate, series } from "./theme.ts";

const MODEL_LABELS: Record<ModelId, string> = {
  majority: "Majority baseline",
  logistic: "Logistic regression",
  hgb: "Gradient boosting",
  rf: "Random forest",
};

const FEATURE_SET_LABELS: Record<FeatureSetId, string> = {
  enrolment: "At enrolment",
  after_s1: "After first semester (leaks)",
};

/** Model colours that stay clear of orange and aqua: those two tokens carry a
 * light-mode contrast warning elsewhere on the page, and a six-line reliability
 * diagram has no room for the extra value labels that warning would call for. */
function modelColor(c: ReturnType<typeof series>, model: ModelId): string {
  if (model === "logistic") return c.s1;
  if (model === "hgb") return c.s4;
  return c.accent;
}

function curveName(curve: CalibrationCurve): string {
  const base = MODEL_LABELS[curve.model];
  return curve.leaks ? `${base} (leaks)` : base;
}

/** Mean absolute gap between predicted and observed, across a curve's bins. */
function meanGap(curve: CalibrationCurve): number {
  const gaps = curve.points.map((p) => Math.abs(p.mean_predicted - p.fraction_positive));
  return gaps.reduce((a, b) => a + b, 0) / gaps.length;
}

/** Below this mean gap, in probability units, the curve reads as well
 * calibrated rather than poorly calibrated. */
const WELL_CALIBRATED_GAP = 0.05;

function bestEnrolmentCurve(data: Calibration): CalibrationCurve | undefined {
  const enrolment = data.curves.filter((c) => c.feature_set === "enrolment");
  return enrolment.slice().sort((a, b) => a.brier - b.brier)[0];
}

function specFor(data: Calibration): FigureSpec {
  const best = bestEnrolmentCurve(data);
  const nTest = data.curves[0]?.points.reduce((a, p) => a + p.n, 0) ?? 0;

  const title = best
    ? `The enrolment models are ${meanGap(best) < WELL_CALIBRATED_GAP ? "well" : "poorly"} calibrated: predicted and observed dropout agree within ${(meanGap(best) * 100).toFixed(1)} points on average for ${MODEL_LABELS[best.model].toLowerCase()}`
    : "The enrolment models' calibration could not be summarised";

  const rows: (string | number)[][] = [];
  for (const curve of data.curves) {
    for (const p of curve.points) {
      rows.push([
        FEATURE_SET_LABELS[curve.feature_set],
        MODEL_LABELS[curve.model],
        pct(p.mean_predicted),
        pct(p.fraction_positive),
        int(p.n),
      ]);
    }
  }

  return {
    id: "fig-calibration",
    title,
    subtitle: `${int(data.bins)} quantile bins per curve over ${int(nTest)} test students; the after-first-semester feature set (dashed curves) leaks information recorded during the course`,
    note: "Source: pipeline test-set predictions; Brier score (lower is better) is in the hover for each curve.",
    alt: `Reliability diagram. Six curves — logistic regression, gradient boosting and random forest, each at enrolment and after the first semester — plot predicted against observed dropout share across ${int(data.bins)} bins, against a diagonal reference. ${title}.`,
    table: {
      columns: ["Feature set", "Model", "Predicted", "Observed", "n"],
      rows,
    },
  };
}

function curveTrace(curve: CalibrationCurve, color: string): Partial<Plotly.PlotData> {
  return {
    type: "scatter",
    mode: "lines+markers",
    name: curveName(curve),
    x: curve.points.map((p) => p.mean_predicted),
    y: curve.points.map((p) => p.fraction_positive),
    customdata: curve.points.map((p) => [int(p.n), num(curve.brier, 3)]),
    hovertemplate:
      "Predicted <b>%{x:.1%}</b>, observed <b>%{y:.1%}</b><br>n = %{customdata[0]}<br>Brier %{customdata[1]}<extra>" +
      curveName(curve) +
      "</extra>",
    line: { color, width: 2, dash: curve.leaks ? "dash" : "solid" },
    marker: { color, size: 9 },
  };
}

export async function render(container: HTMLElement): Promise<void> {
  let data: Calibration;
  try {
    data = await loadJson<Calibration>("calibration.json");
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the calibration data.",
      () => void render(container),
    );
    return;
  }

  if (!data.curves.length) {
    showError(container, "The calibration file carries no curves.", () =>
      void render(container),
    );
    return;
  }

  // Half the curves are the after-first-semester series, always on screen.
  container.classList.add("leaks");

  const plot = mountFigure(container, specFor(data));

  function draw(): Promise<unknown> {
    const c = series();
    const grid = cssVar("--grid");
    const traces: Partial<Plotly.PlotData>[] = [
      {
        type: "scatter",
        mode: "lines",
        name: "Perfect calibration",
        x: [0, 1],
        y: [0, 1],
        line: { color: grid, width: 1, dash: "dot" },
        hoverinfo: "skip",
        showlegend: false,
      },
      ...data.curves.map((curve) => curveTrace(curve, modelColor(c, curve.model))),
    ];
    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: true,
      legend: horizontalLegend(),
      margin: { ...base.margin, t: 30 },
      hovermode: "closest",
      xaxis: {
        ...base.xaxis,
        range: [0, 1],
        tickformat: ".0%",
        title: { text: "Mean predicted probability of dropout", standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        range: [0, 1],
        tickformat: ".0%",
        title: { text: "Observed share who dropped out", standoff: 8 },
      },
    };
    return Plotly.react(plot, traces, layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
