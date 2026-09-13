import { loadJson } from "../data.ts";
import {
  controlSlot,
  mountFigure,
  showError,
  updateFigure,
  type FigureSpec,
} from "../figure.ts";
import { int, lowerLead, num } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { onThemeChange } from "../theme.ts";
import type {
  FeatureSetId,
  ImportanceResult,
  ModelId,
  PermutationImportance,
  Target,
} from "../types.ts";
import { CONFIG, errorBars, layoutTemplate, reversed, series } from "./theme.ts";

const MODEL_LABELS: Record<ModelId, string> = {
  majority: "Majority baseline",
  logistic: "Logistic regression",
  hgb: "Gradient boosting",
  rf: "Random forest",
};

const TARGET_LABELS: Record<Target, string> = {
  binary: "Dropout or not",
  multiclass: "Dropout, Enrolled or Graduate",
};

const FEATURE_SET_LABELS: Record<FeatureSetId, string> = {
  enrolment: "at enrolment",
  after_s1: "after the first semester (leaks)",
};

function resultKey(r: ImportanceResult): string {
  return `${r.target}|${r.feature_set}`;
}

function optionLabel(r: ImportanceResult): string {
  return `${TARGET_LABELS[r.target]}, ${FEATURE_SET_LABELS[r.feature_set]}`;
}

function specFor(result: ImportanceResult): FigureSpec {
  const top = result.features[0];
  const phrase = result.feature_set === "enrolment" ? "at enrolment" : "after the first semester";
  const title = top
    ? `${top.label} matters most to the ${lowerLead(MODEL_LABELS[result.model])} model ${phrase}`
    : `Permutation importance carries no features for the ${lowerLead(MODEL_LABELS[result.model])} model ${phrase}`;

  const rows: (string | number)[][] = result.features.map((f) => [
    f.label,
    num(f.mean, 4),
    num(f.sd, 4),
  ]);

  return {
    id: "fig-importance",
    title,
    subtitle: `${TARGET_LABELS[result.target]}, ${FEATURE_SET_LABELS[result.feature_set]}; ${lowerLead(MODEL_LABELS[result.model])}, baseline score ${num(result.baseline_score, 3)}`,
    note: `Ten repeats, seed 20260912; a variable is permuted as a whole, before encoding. A shuffled macro indicator (unemployment, inflation, GDP) is a cohort effect, not an economic one.`,
    alt: `Horizontal bar chart of permutation importance for ${int(result.features.length)} variables, mean score drop with a standard-deviation whisker, ${TARGET_LABELS[result.target]} ${FEATURE_SET_LABELS[result.feature_set]}. ${title}.`,
    table: {
      columns: ["Variable", "Mean score drop", "SD"],
      rows,
    },
  };
}

function barTrace(result: ImportanceResult, color: string): Partial<Plotly.PlotData> {
  const fs = reversed(result.features);
  return {
    type: "bar",
    orientation: "h",
    y: fs.map((f) => f.label),
    x: fs.map((f) => f.mean),
    text: fs.map((f) => num(f.mean, 3)),
    textposition: "outside",
    marker: { color },
    hovertemplate: "<b>%{y}</b><br>mean %{x:.4f} ± %{customdata:.4f}<extra></extra>",
    customdata: fs.map((f) => f.sd),
    error_x: errorBars(
      fs.map((f) => f.mean),
      fs.map((f) => f.mean - f.sd),
      fs.map((f) => f.mean + f.sd),
      color,
    ),
  };
}

function resultPicker(
  results: ImportanceResult[],
  selected: string,
  onChange: (key: string) => void,
): HTMLDivElement {
  const wrap = document.createElement("div");
  wrap.className = "fig-picker";
  const select = document.createElement("select");
  select.id = "fig-importance-picker";
  for (const r of results) {
    const opt = document.createElement("option");
    opt.value = resultKey(r);
    opt.textContent = optionLabel(r);
    opt.selected = resultKey(r) === selected;
    select.appendChild(opt);
  }
  select.addEventListener("change", () => onChange(select.value));
  const label = document.createElement("label");
  label.htmlFor = select.id;
  label.textContent = "Model and feature set";
  wrap.append(label, select);
  return wrap;
}

export async function render(container: HTMLElement): Promise<void> {
  let data: PermutationImportance;
  try {
    data = await loadJson<PermutationImportance>("permutation_importance.json");
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the importance data.",
      () => void render(container),
    );
    return;
  }

  const first = data.results[0];
  if (!first) {
    showError(container, "The importance file carries no results.", () =>
      void render(container),
    );
    return;
  }
  let result: ImportanceResult = first;
  container.classList.toggle("leaks", result.leaks);

  const plot = mountFigure(container, specFor(result));
  controlSlot(plot).before(
    resultPicker(data.results, resultKey(result), (key) => {
      const next = data.results.find((r) => resultKey(r) === key);
      if (!next) return;
      result = next;
      container.classList.toggle("leaks", result.leaks);
      updateFigure(container, specFor(result));
      void draw();
    }),
  );

  function draw(): Promise<unknown> {
    const c = series();
    const trace = barTrace(result, c.s1);
    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: false,
      margin: { ...base.margin, t: 12, r: 56 },
      hovermode: "closest",
      xaxis: { ...base.xaxis, zeroline: true, title: { text: "Mean drop in score", standoff: 8 } },
      yaxis: {
        ...base.yaxis,
        automargin: true,
        showgrid: false,
        categoryorder: "array",
        categoryarray: reversed(result.features.map((f) => f.label)),
      },
    };
    return Plotly.react(plot, [trace], layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
