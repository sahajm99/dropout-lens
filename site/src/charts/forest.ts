import { loadJson } from "../data.ts";
import {
  controlSlot,
  mountFigure,
  showError,
  updateFigure,
  type FigureSpec,
} from "../figure.ts";
import { int, lowerLead, pval, ratio } from "../fmt.ts";
import Plotly from "../plotly.ts";
import { onThemeChange } from "../theme.ts";
import type { BinaryOddsModel, OddsRatios, OddsTerm } from "../types.ts";
import { CONFIG, cssVar, layoutTemplate, reversed, series } from "./theme.ts";

type ModelKey = "binary_enrolment" | "binary_after_s1";

const VIEWS: { key: ModelKey; button: string }[] = [
  { key: "binary_enrolment", button: "At enrolment" },
  { key: "binary_after_s1", button: "After first semester (leaks)" },
];

function pText(p: number | null): string {
  if (p === null) return "";
  return p < 0.001 ? `p ${pval(p)}` : `p = ${pval(p)}`;
}

function ciRatio(lo: number | null, hi: number | null): string {
  if (lo === null || hi === null) return "—";
  return `${ratio(lo)} to ${ratio(hi)}`;
}

/** A term the file gives a finite interval for, whether or not it is flagged
 * unstable: the Biofuel Production Technologies row is drawn muted, never
 * skipped, in the binary models this figure reads. */
function hasCi(t: OddsTerm): boolean {
  return t.is_reference || (typeof t.lo === "number" && typeof t.hi === "number");
}

/** The non-reference, non-unstable term furthest from an odds ratio of 1, on
 * the log scale the chart itself uses: the one claim the title can make
 * without also having to explain away an unreliable estimate. */
function strongestTerm(terms: OddsTerm[]): OddsTerm | undefined {
  const candidates = terms.filter((t) => !t.is_reference && !t.unstable && hasCi(t));
  if (!candidates.length) return undefined;
  return candidates.reduce((a, b) =>
    Math.abs(Math.log(b.or)) > Math.abs(Math.log(a.or)) ? b : a,
  );
}

function specFor(model: BinaryOddsModel): FigureSpec {
  const skipped = model.terms.filter((t) => !hasCi(t));
  const strongest = strongestTerm(model.terms);
  const phrase = model.leaks ? "after the first semester" : "at enrolment";

  const title = strongest
    ? `Holding the rest constant, ${lowerLead(strongest.label)} multiplies the odds of leaving by ${ratio(strongest.or)}`
    : `Adjusted odds ratios for the model ${phrase}`;

  const rows: (string | number)[][] = model.terms.map((t) => [
    t.variable,
    t.label,
    t.is_reference ? "1 (reference)" : ratio(t.or),
    ciRatio(t.lo, t.hi),
    t.unstable ? "unstable: too few students to estimate reliably" : "",
    int(t.n_level),
  ]);

  return {
    id: "fig-forest",
    title,
    subtitle: `Adjusted odds ratios with 95% intervals, logistic regression on ${int(model.n_obs)} training students, ${phrase}${model.leaks ? "; this view leaks information recorded during the course" : ""}`,
    note:
      skipped.length > 0
        ? `Source: pipeline logistic fit, train rows. ${int(skipped.length)} level${skipped.length === 1 ? "" : "s"} too few students to estimate is not drawn; a muted marker is drawn but flagged unstable in the table.`
        : "Source: pipeline logistic fit, train rows. A muted marker is drawn but flagged unstable in the table where a level's training rows almost all share one outcome.",
    alt: `Forest plot on a log scale of ${int(model.terms.length)} logistic-regression terms, one row per level of each covariate, reference levels hollow at an odds ratio of 1. ${title}.`,
    table: {
      columns: ["Variable", "Level", "Odds ratio", "95% CI", "Flag", "Training rows"],
      rows,
    },
  };
}

const HOVER =
  "<b>%{x:.2f}x</b> %{customdata[0]}<br>n = %{customdata[1]:,}%{customdata[2]}<extra>%{y}</extra>";

function termTrace(
  terms: OddsTerm[],
  color: string,
  open: boolean,
  bars: boolean,
): Partial<Plotly.PlotData> {
  const ts = reversed(terms);
  return {
    type: "scatter",
    mode: "markers",
    showlegend: false,
    y: ts.map((t) => t.label),
    x: ts.map((t) => t.or),
    customdata: ts.map((t) => [
      t.is_reference ? "(reference level)" : `(95% CI ${ciRatio(t.lo, t.hi)})`,
      t.n_level,
      t.p_value === null ? "" : `<br>${pText(t.p_value)}`,
    ]),
    hovertemplate: HOVER,
    marker: {
      symbol: open ? "circle-open" : "circle",
      color,
      size: 7,
      line: { color, width: 1.5 },
    },
    error_x: bars
      ? {
          type: "data",
          symmetric: false,
          array: ts.map((t) => (t.hi ?? t.or) - t.or),
          arrayminus: ts.map((t) => t.or - (t.lo ?? t.or)),
          color,
          thickness: 1,
          width: 0,
        }
      : undefined,
  };
}

function traces(model: BinaryOddsModel, color: string, muted: string): Partial<Plotly.PlotData>[] {
  const drawable = model.terms.filter(hasCi);
  const refs = drawable.filter((t) => t.is_reference);
  const stable = drawable.filter((t) => !t.is_reference && !t.unstable);
  const unstable = drawable.filter((t) => !t.is_reference && t.unstable);
  const out: Partial<Plotly.PlotData>[] = [];
  if (refs.length) out.push(termTrace(refs, color, true, false));
  if (stable.length) out.push(termTrace(stable, color, false, true));
  if (unstable.length) out.push(termTrace(unstable, muted, false, true));
  return out;
}

function toggle(
  selected: ModelKey,
  onChange: (key: ModelKey) => void,
): HTMLDivElement {
  const group = document.createElement("div");
  group.className = "seg";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Feature set");
  for (const view of VIEWS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "seg-btn";
    btn.textContent = view.button;
    btn.setAttribute("aria-pressed", String(view.key === selected));
    btn.addEventListener("click", () => {
      for (const other of group.querySelectorAll("button")) {
        other.setAttribute("aria-pressed", String(other === btn));
      }
      onChange(view.key);
    });
    group.appendChild(btn);
  }
  return group;
}

export async function render(container: HTMLElement): Promise<void> {
  let data: OddsRatios;
  try {
    data = await loadJson<OddsRatios>("odds_ratios.json");
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the model data.",
      () => void render(container),
    );
    return;
  }

  let key: ModelKey = "binary_enrolment";
  let model = data.models[key];
  container.classList.toggle("leaks", model.leaks);

  const plot = mountFigure(container, specFor(model));
  controlSlot(plot).before(
    toggle(key, (next) => {
      key = next;
      model = data.models[key];
      container.classList.toggle("leaks", model.leaks);
      updateFigure(container, specFor(model));
      void draw();
    }),
  );

  function draw(): Promise<unknown> {
    const c = series();
    const ink = cssVar("--ink-3");
    const rows = traces(model, c.s1, ink);
    const drawable = model.terms.filter(hasCi);
    const values = drawable.flatMap((t) => [t.or, t.lo ?? t.or, t.hi ?? t.or]);
    const xRange: [number, number] = [
      Math.log10(Math.min(...values) / 1.25),
      Math.log10(Math.max(...values) * 1.25),
    ];
    const ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50].filter(
      (t) => t >= 10 ** xRange[0] && t <= 10 ** xRange[1],
    );
    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      showlegend: false,
      hovermode: "closest",
      margin: { ...base.margin, t: 12, b: 44 },
      xaxis: {
        ...base.xaxis,
        type: "log",
        range: xRange,
        tickmode: "array",
        tickvals: ticks,
        ticktext: ticks.map((t) => `${t}x`),
        title: { text: "Adjusted odds ratio (log scale)", standoff: 6 },
      },
      yaxis: {
        ...base.yaxis,
        automargin: true,
        showgrid: false,
        categoryorder: "array",
        categoryarray: reversed(drawable.map((t) => t.label)),
        tickfont: { color: cssVar("--ink-2"), size: 11 },
      },
      shapes: [
        {
          type: "line",
          xref: "x",
          yref: "paper",
          x0: 1,
          x1: 1,
          y0: 0,
          y1: 1,
          line: { color: cssVar("--ink-3"), width: 1, dash: "dot" },
        },
      ],
    };
    return Plotly.react(plot, rows, layout, CONFIG);
  }

  onThemeChange(() => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
