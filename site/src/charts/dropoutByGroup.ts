import { loadJson } from "../data.ts";
import {
  controlSlot,
  mountFigure,
  showError,
  updateFigure,
  type FigureSpec,
} from "../figure.ts";
import { ciText, int, lowerLead, pct, upperLead } from "../fmt.ts";
import Plotly from "../plotly.ts";
import type { Group, OutcomeByGroup } from "../types.ts";
import {
  CONFIG,
  cssVar,
  errorBars,
  layoutTemplate,
  outcomeColor,
  reversed,
} from "./theme.ts";

/** The nine groups known at enrolment; the approval band gets its own figure. */
function enrolmentGroups(data: OutcomeByGroup): Group[] {
  return data.groups.filter((g) => !g.post_enrolment);
}

interface Point {
  level: string;
  p: number;
  lo: number;
  hi: number;
  n: number;
  smallN: boolean;
}

/** A suppressed level carries no Dropout share and is left out, not zeroed. */
function points(group: Group): Point[] {
  const out: Point[] = [];
  for (const r of group.rows) {
    if (r.suppressed || !r.shares) continue;
    const s = r.shares.Dropout;
    out.push({
      level: r.level,
      p: s.p,
      lo: s.lo,
      hi: s.hi,
      n: r.n,
      smallN: r.small_n,
    });
  }
  return out;
}

function specFor(data: OutcomeByGroup, group: Group): FigureSpec {
  const pts = points(group);
  let title: string;
  if (pts.length >= 2) {
    const worst = pts.reduce((a, b) => (b.p > a.p ? b : a));
    const best = pts.reduce((a, b) => (b.p < a.p ? b : a));
    title = `${upperLead(worst.level)} students left at ${pct(worst.p)}, against ${pct(best.p)} for ${lowerLead(best.level)}`;
  } else if (pts.length === 1) {
    title = `${upperLead(pts[0]!.level)} students left at ${pct(pts[0]!.p)}`;
  } else {
    title = `Every level of ${lowerLead(group.label)} is suppressed`;
  }
  return {
    id: "fig-dropout-by-group",
    title,
    subtitle: `Dropout share by ${lowerLead(group.label)}, ${int(data.n_total)} students; 95% Wilson intervals; grey marks a level under ${int(data.small_n_below)}`,
    note: `Source: Realinho, Machado, Baptista and Martins, UCI Machine Learning Repository. Cells under ${int(data.suppress_below)} students are suppressed and left out of the chart.`,
    alt: `Dot plot of the Dropout share for each level of ${lowerLead(group.label)}, with 95% Wilson intervals. ${title}.`,
    table: {
      columns: ["Level", "Students", "Dropout share", "95% CI"],
      rows: group.rows.map((r) => {
        const s = r.suppressed || !r.shares ? null : r.shares.Dropout;
        return [
          r.level,
          int(r.n) + (r.small_n ? " (thin)" : ""),
          s ? pct(s.p) : "suppressed",
          s ? ciText(s.lo, s.hi) : "—",
        ];
      }),
    },
  };
}

function picker(
  groups: Group[],
  selected: string,
  onChange: (id: string) => void,
): HTMLDivElement {
  const wrap = document.createElement("div");
  wrap.className = "fig-picker";
  const select = document.createElement("select");
  select.id = "fig-dropout-by-group-picker";
  for (const g of groups) {
    const opt = document.createElement("option");
    opt.value = g.id;
    opt.textContent = g.label;
    opt.selected = g.id === selected;
    select.appendChild(opt);
  }
  select.addEventListener("change", () => onChange(select.value));
  const label = document.createElement("label");
  label.htmlFor = select.id;
  label.textContent = "Group";
  wrap.append(label, select);
  return wrap;
}

const HOVER =
  "%{y}<br><b>%{x:.1%}</b> (95% CI %{customdata[0]:.1%} to %{customdata[1]:.1%})<br>n = %{customdata[2]:,}<extra></extra>";

export async function render(container: HTMLElement): Promise<void> {
  let data: OutcomeByGroup;
  try {
    data = await loadJson<OutcomeByGroup>("outcome_by_group.json");
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the group data.",
      () => void render(container),
    );
    return;
  }

  const groups = enrolmentGroups(data);
  const first = groups[0];
  if (!first) {
    showError(
      container,
      "The group file carries no enrolment-time groups.",
      () => void render(container),
    );
    return;
  }
  let group: Group = first;

  const plot = mountFigure(container, specFor(data, group));
  controlSlot(plot).before(
    picker(groups, group.id, (id) => {
      const next = groups.find((g) => g.id === id);
      if (!next) return;
      group = next;
      updateFigure(container, specFor(data, group));
      void draw();
    }),
  );

  function draw(): Promise<unknown> {
    const pts = reversed(points(group));
    const color = outcomeColor("Dropout");
    const muted = cssVar("--ink-3");
    const errColor = cssVar("--ink-2");
    const trace: Partial<Plotly.PlotData> = {
      type: "scatter",
      mode: "markers",
      name: "Dropout share",
      y: pts.map((d) => d.level),
      x: pts.map((d) => d.p),
      customdata: pts.map((d) => [d.lo, d.hi, d.n]),
      hovertemplate: HOVER,
      marker: {
        color: pts.map((d) => (d.smallN ? muted : color)),
        size: 10,
        line: { color: errColor, width: 1 },
      },
      error_x: errorBars(
        pts.map((d) => d.p),
        pts.map((d) => d.lo),
        pts.map((d) => d.hi),
        errColor,
      ),
      showlegend: false,
    };

    const shown = new Set(pts.map((d) => d.level));
    const annotations: Partial<Plotly.Annotations>[] = group.levels
      .filter((lvl) => !shown.has(lvl))
      .map((lvl) => ({
        x: 0,
        y: lvl,
        xanchor: "left",
        text: "suppressed",
        showarrow: false,
        font: { color: muted, size: 11 },
      }));

    const maxHi = Math.max(0, ...pts.map((d) => d.hi));
    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      annotations,
      showlegend: false,
      margin: { ...base.margin, t: 12 },
      hovermode: "closest",
      xaxis: {
        ...base.xaxis,
        tickformat: ".0%",
        range: [0, maxHi + 0.04],
        title: { text: "Dropout share", standoff: 8 },
      },
      yaxis: {
        ...base.yaxis,
        type: "category",
        automargin: true,
        showgrid: false,
        categoryorder: "array",
        categoryarray: reversed(group.levels),
      },
    };
    return Plotly.react(plot, [trace], layout, CONFIG);
  }

  document.addEventListener("themechange", () => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
