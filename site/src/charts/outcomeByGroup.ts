import { loadJson } from "../data.ts";
import {
  controlSlot,
  mountFigure,
  showError,
  updateFigure,
  type FigureSpec,
} from "../figure.ts";
import { int, lowerLead, pct } from "../fmt.ts";
import Plotly from "../plotly.ts";
import type { Group, GroupRow, Outcome, OutcomeByGroup } from "../types.ts";
import {
  CONFIG,
  OUTCOMES,
  cssVar,
  horizontalLegend,
  inkOn,
  layoutTemplate,
  outcomeColor,
  reversed,
} from "./theme.ts";

/** The nine groups known at enrolment; the approval band gets its own figure. */
function enrolmentGroups(data: OutcomeByGroup): Group[] {
  return data.groups.filter((g) => !g.post_enrolment);
}

function rowShare(row: GroupRow, outcome: Outcome): number | null {
  return row.suppressed || !row.shares ? null : row.shares[outcome].p;
}

function dropoutShare(row: GroupRow): number | null {
  return rowShare(row, "Dropout");
}

/** The level with the widest and narrowest Dropout share, suppressed rows excluded. */
function bounds(group: Group): { min: GroupRow; max: GroupRow } | null {
  const usable = group.rows.filter((r) => dropoutShare(r) !== null);
  if (!usable.length) return null;
  const min = usable.reduce((a, b) =>
    (dropoutShare(b) as number) < (dropoutShare(a) as number) ? b : a,
  );
  const max = usable.reduce((a, b) =>
    (dropoutShare(b) as number) > (dropoutShare(a) as number) ? b : a,
  );
  return { min, max };
}

function specFor(data: OutcomeByGroup, group: Group): FigureSpec {
  const b = bounds(group);
  const title = b
    ? `Among ${lowerLead(group.label)}, dropout runs from ${pct(dropoutShare(b.min) as number)} (${b.min.level}) to ${pct(dropoutShare(b.max) as number)} (${b.max.level})`
    : `Among ${lowerLead(group.label)}, every level is suppressed`;
  return {
    id: "fig-outcome-by-group",
    title,
    subtitle: `Outcome shares by ${lowerLead(group.label)}; ${int(data.n_total)} students; 95% Wilson intervals in the table`,
    note: `Source: Realinho, Machado, Baptista and Martins, UCI Machine Learning Repository. Cells under ${int(data.suppress_below)} students are suppressed.`,
    alt: `Horizontal 100% stacked bar chart of dropout, still-enrolled and graduate shares for each level of ${lowerLead(group.label)}. ${title}.`,
    table: {
      columns: ["Level", "Students", "Dropout", "Enrolled", "Graduate"],
      rows: group.rows.map((r) => [
        r.level,
        int(r.n),
        r.suppressed || !r.shares ? "suppressed" : pct(r.shares.Dropout.p),
        r.suppressed || !r.shares ? "suppressed" : pct(r.shares.Enrolled.p),
        r.suppressed || !r.shares ? "suppressed" : pct(r.shares.Graduate.p),
      ]),
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
  select.id = "fig-outcome-by-group-picker";
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
  "<b>%{fullData.name}</b>: %{x:.1%} of %{y}<br>95% CI %{customdata[0]:.1%} to %{customdata[1]:.1%}<br>n = %{customdata[2]:,}<extra></extra>";

function outcomeTrace(
  rows: GroupRow[],
  outcome: Outcome,
  color: string,
  gapColor: string,
): Partial<Plotly.PlotData> {
  const rs = reversed(rows);
  const shares = rs.map((r) => rowShare(r, outcome));
  return {
    type: "bar",
    orientation: "h",
    name: outcome,
    y: rs.map((r) => r.level),
    x: shares,
    customdata: rs.map((r) => {
      const s = r.shares;
      return s ? [s[outcome].lo, s[outcome].hi, r.n] : [null, null, r.n];
    }),
    text: shares.map((p) => (p !== null && p >= 0.08 ? pct(p, 0) : "")),
    textposition: "inside",
    insidetextanchor: "middle",
    textfont: { color: inkOn(color) },
    marker: { color, line: { color: gapColor, width: 2 } },
    hovertemplate: HOVER,
  };
}

/** One full-width grey hatched bar per suppressed level, labelled from the data. */
function suppressedTrace(
  data: OutcomeByGroup,
  rows: GroupRow[],
  color: string,
  gapColor: string,
): Partial<Plotly.PlotData> | null {
  const rs = reversed(rows);
  if (!rs.some((r) => r.suppressed)) return null;
  const label = `suppressed (n under ${int(data.suppress_below)})`;
  return {
    type: "bar",
    orientation: "h",
    name: "Suppressed",
    y: rs.map((r) => r.level),
    x: rs.map((r) => (r.suppressed ? 1 : null)),
    text: rs.map((r) => (r.suppressed ? label : "")),
    textposition: "inside",
    insidetextanchor: "middle",
    textfont: { color: inkOn(color) },
    marker: {
      color,
      pattern: { shape: "/", fgcolor: gapColor, size: 6, solidity: 0.35 },
      line: { color: gapColor, width: 2 },
    },
    hovertemplate: `%{y}: ${label}<extra></extra>`,
    showlegend: false,
  };
}

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
    const gapColor = cssVar("--surface");
    const mutedColor = cssVar("--ink-3");
    const traces = [
      ...OUTCOMES.map((o) => outcomeTrace(group.rows, o, outcomeColor(o), gapColor)),
      suppressedTrace(data, group.rows, mutedColor, gapColor),
    ].filter((t): t is Partial<Plotly.PlotData> => t !== null);
    const base = layoutTemplate();
    const layout: Partial<Plotly.Layout> = {
      ...base,
      barmode: "stack",
      bargap: 0.25,
      showlegend: true,
      legend: horizontalLegend(),
      margin: { ...base.margin, t: 30 },
      hovermode: "closest",
      xaxis: {
        ...base.xaxis,
        tickformat: ".0%",
        range: [0, 1.02],
        title: { text: "Share of students", standoff: 8 },
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
    return Plotly.react(plot, traces, layout, CONFIG);
  }

  document.addEventListener("themechange", () => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
