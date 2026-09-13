import { loadJson } from "../data.ts";
import { mountFigure, showError, type FigureSpec } from "../figure.ts";
import { int, pct } from "../fmt.ts";
import Plotly from "../plotly.ts";
import type { GroupRow, Outcome, OutcomeByGroup } from "../types.ts";
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

const GROUP_ID = "approval_band_s1";
const NOTHING_LEVEL = "0% approved";
const EVERYTHING_LEVEL = "100% approved";

function rowShare(row: GroupRow, outcome: Outcome): number | null {
  return row.suppressed || !row.shares ? null : row.shares[outcome].p;
}

function specFor(data: OutcomeByGroup): FigureSpec | null {
  const group = data.groups.find((g) => g.id === GROUP_ID);
  if (!group) return null;
  const nothing = group.rows.find((r) => r.level === NOTHING_LEVEL);
  const everything = group.rows.find((r) => r.level === EVERYTHING_LEVEL);
  const nothingP = nothing ? rowShare(nothing, "Dropout") : null;
  const everythingP = everything ? rowShare(everything, "Dropout") : null;
  const title =
    nothingP !== null && everythingP !== null
      ? `Students who passed nothing in the first semester left at ${pct(nothingP)}; those who passed everything, at ${pct(everythingP)}`
      : "First-semester approval band and the outcome that followed it";
  return {
    id: "fig-approval-band",
    title,
    subtitle: `${group.label}, ${int(data.n_total)} students; leaks: this is a first-semester result, not an enrolment fact`,
    note: "Source: Realinho, Machado, Baptista and Martins, UCI Machine Learning Repository. “No units enrolled” is kept as its own band, not counted as 0% approved.",
    alt: `Horizontal 100% stacked bar chart of dropout, still-enrolled and graduate shares for each first-semester approval band. ${title}.`,
    table: {
      columns: ["Approval band", "Students", "Dropout", "Enrolled", "Graduate"],
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

  const spec = specFor(data);
  const group = data.groups.find((g) => g.id === GROUP_ID);
  if (!spec || !group) {
    showError(
      container,
      "The group file carries no first-semester approval band.",
      () => void render(container),
    );
    return;
  }

  const plot = mountFigure(container, spec);

  function draw(): Promise<unknown> {
    const gapColor = cssVar("--surface");
    const traces = OUTCOMES.map((o) =>
      outcomeTrace(group!.rows, o, outcomeColor(o), gapColor),
    );
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
        categoryarray: reversed(group!.levels),
      },
    };
    return Plotly.react(plot, traces, layout, CONFIG);
  }

  document.addEventListener("themechange", () => {
    if (plot.isConnected) void draw();
  });
  await draw();
}
