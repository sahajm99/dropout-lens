import { loadJson } from "../data.ts";
import {
  controlSlot,
  mountFigure,
  showError,
  updateFigure,
  type FigureSpec,
} from "../figure.ts";
import { ciText, int, lowerLead, num, pct, pct0, upperLead } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { cssVar, onThemeChange } from "../theme.ts";
import type {
  FeatureSetId,
  MetricValue,
  ModelId,
  ModelMetrics,
  ModelResult,
} from "../types.ts";
import {
  CONFIG,
  errorBars,
  horizontalLegend,
  layoutTemplate,
  reversed,
  series,
} from "./theme.ts";

type MetricKey =
  | "accuracy"
  | "balanced_accuracy"
  | "macro_f1"
  | "roc_auc"
  | "pr_auc";

interface MetricDef {
  key: MetricKey;
  label: string;
  fmt: (v: number) => string;
}

/** Macro F1 is the default here: the metric the pipeline selects models on,
 * and the fairest single number once the classes are this imbalanced. */
const METRICS: MetricDef[] = [
  { key: "accuracy", label: "accuracy", fmt: (v) => pct(v) },
  { key: "balanced_accuracy", label: "balanced accuracy", fmt: (v) => pct(v) },
  { key: "macro_f1", label: "macro F1", fmt: (v) => pct(v) },
  { key: "roc_auc", label: "ROC AUC", fmt: (v) => num(v) },
  { key: "pr_auc", label: "PR AUC", fmt: (v) => num(v) },
];
const DEFAULT_METRIC_KEY: MetricKey = "macro_f1";

const FEATURE_SETS: FeatureSetId[] = ["enrolment", "after_s1"];

function findResult(
  data: ModelMetrics,
  fs: FeatureSetId,
  model: ModelId,
): ModelResult | undefined {
  return data.results.find(
    (r) => r.target === "multiclass" && r.feature_set === fs && r.model === model,
  );
}

function legendName(fs: FeatureSetId): string {
  return fs === "after_s1" ? "After first semester (leaks)" : "At enrolment";
}

/** The class every majority-baseline row is predicted as, read off its own
 * confusion matrix rather than assumed, so a re-run that flips the mode
 * outcome is still described correctly. */
function majorityClass(data: ModelMetrics): string {
  const classes = data.targets.multiclass.classes;
  const baseline = findResult(data, "enrolment", "majority");
  const row = baseline?.confusion[0];
  const idx = row?.findIndex((v) => v > 0) ?? -1;
  return (idx >= 0 ? classes[idx] : classes[0]) ?? "Graduate";
}

function specFor(data: ModelMetrics, metric: MetricDef): FigureSpec {
  const bestEnrolModel = data.best.multiclass.enrolment;
  const bestAfterModel = data.best.multiclass.after_s1;
  const rEnrol = findResult(data, "enrolment", bestEnrolModel);
  const rAfter = findResult(data, "after_s1", bestAfterModel);
  const vEnrol = rEnrol?.test[metric.key] ?? null;
  const vAfter = rAfter?.test[metric.key] ?? null;

  const title =
    vEnrol && vAfter
      ? `At enrolment the best model reaches ${metric.fmt(vEnrol.value)} ${metric.label} on the three-way outcome (${lowerLead(data.model_labels[bestEnrolModel])}); after the first semester, ${metric.fmt(vAfter.value)}`
      : `At enrolment the best three-way model is ${lowerLead(data.model_labels[bestEnrolModel])}; after the first semester it is ${lowerLead(data.model_labels[bestAfterModel])}`;

  const undefinedForBaseline = metric.key === "roc_auc" || metric.key === "pr_auc";

  const rows: (string | number)[][] = [];
  for (const model of data.models) {
    for (const fs of FEATURE_SETS) {
      const r = findResult(data, fs, model);
      const v = r?.test[metric.key] ?? null;
      rows.push([
        data.model_labels[model],
        data.feature_sets[fs].label,
        v ? metric.fmt(v.value) : "not defined for the baseline",
        v ? ciText(v.lo, v.hi) : "—",
      ]);
    }
  }

  return {
    id: "fig-metrics-multiclass",
    title,
    subtitle: `Test set of ${int(data.split.n_test)} students (${int(data.split.n_train)} train), ${pct0(data.bootstrap.level)} bootstrap intervals over ${int(data.bootstrap.resamples)} resamples; the after-first-semester series leaks information recorded during the course`,
    note: `Train ${int(data.split.n_train)}, test ${int(data.split.n_test)}, split stratified on ${data.split.stratified_on}; the majority baseline always predicts "${majorityClass(data)}".${undefinedForBaseline ? " ROC AUC and PR AUC are not defined for that baseline, so it is left off this view." : ""}`,
    alt: `Dot plot of four models' ${metric.label} on the Dropout, Enrolled or Graduate outcome, at enrolment and after the first semester, with bootstrap intervals. ${title}.`,
    table: {
      columns: ["Model", "Feature set", upperLead(metric.label), "95% CI"],
      rows,
    },
  };
}

function dotTrace(
  data: ModelMetrics,
  fs: FeatureSetId,
  metric: MetricDef,
  color: string,
  withLabels: boolean,
  labelColor: string,
): Partial<Plotly.PlotData> {
  const pts = data.models
    .map((model) => {
      const r = findResult(data, fs, model);
      const v = r?.test[metric.key] ?? null;
      return v ? { label: data.model_labels[model], v } : null;
    })
    .filter((p): p is { label: string; v: MetricValue } => p !== null);
  const ordered = reversed(pts);
  return {
    type: "scatter",
    mode: withLabels ? "text+markers" : "markers",
    name: legendName(fs),
    y: ordered.map((p) => p.label),
    x: ordered.map((p) => p.v.value),
    text: withLabels ? ordered.map((p) => metric.fmt(p.v.value)) : undefined,
    textposition: "middle right",
    textfont: { color: labelColor, size: 12 },
    customdata: ordered.map((p) => [metric.fmt(p.v.value), ciText(p.v.lo, p.v.hi)]),
    hovertemplate: "%{y}<br><b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
    marker: { color, size: 10 },
    error_x: errorBars(
      ordered.map((p) => p.v.value),
      ordered.map((p) => p.v.lo),
      ordered.map((p) => p.v.hi),
      color,
    ),
  };
}

function metricPicker(
  selected: MetricKey,
  onChange: (key: MetricKey) => void,
): HTMLDivElement {
  const wrap = document.createElement("div");
  wrap.className = "fig-picker";
  const select = document.createElement("select");
  select.id = "fig-metrics-multiclass-picker";
  for (const m of METRICS) {
    const opt = document.createElement("option");
    opt.value = m.key;
    opt.textContent = upperLead(m.label);
    opt.selected = m.key === selected;
    select.appendChild(opt);
  }
  select.addEventListener("change", () => onChange(select.value as MetricKey));
  const label = document.createElement("label");
  label.htmlFor = select.id;
  label.textContent = "Metric";
  wrap.append(label, select);
  return wrap;
}

export async function render(container: HTMLElement): Promise<void> {
  let data: ModelMetrics;
  try {
    data = await loadJson<ModelMetrics>("model_metrics.json");
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the model metrics.",
      () => void render(container),
    );
    return;
  }

  // Both feature sets are always drawn, so the leak marker is permanent.
  container.classList.add("leaks");

  let metric: MetricDef =
    METRICS.find((m) => m.key === DEFAULT_METRIC_KEY) ?? METRICS[0]!;
  const plot = mountFigure(container, specFor(data, metric));
  controlSlot(plot).before(
    metricPicker(metric.key, (key) => {
      const next = METRICS.find((m) => m.key === key);
      if (!next) return;
      metric = next;
      updateFigure(container, specFor(data, metric));
      void draw();
    }),
  );

  function draw(): Promise<unknown> {
    const c = series();
    const ink = cssVar("--ink");
    const isPct = metric.key !== "roc_auc" && metric.key !== "pr_auc";
    const traces = [
      dotTrace(data, "enrolment", metric, c.s1, false, ink),
      dotTrace(data, "after_s1", metric, c.s2, true, ink),
    ];
    const categoryarray = reversed(data.models.map((m) => data.model_labels[m]));
    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: true,
      legend: horizontalLegend(),
      margin: { ...base.margin, t: 30, r: 90 },
      hovermode: "closest",
      xaxis: {
        ...base.xaxis,
        range: [0, 1.08],
        tickformat: isPct ? ".0%" : ".2f",
        title: { text: upperLead(metric.label), standoff: 8 },
      },
      yaxis: { ...base.yaxis, automargin: true, showgrid: false, categoryarray },
    };
    return Plotly.react(plot, traces, layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
