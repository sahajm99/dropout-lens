/**
 * The hero: a ten by ten grid of tiles, one per hundred students, coloured by
 * outcome, and the same split as a slim strip in the rail. Both are built
 * from `outcome_overall.json`, so the picture and the numbers cannot drift.
 */
import { OUTCOMES, outcomeToken } from "./charts/theme.ts";
import { loadJson } from "./data.ts";
import { detailsTable, el, showError } from "./figure.ts";
import { ciText, int, pct, pct0 } from "./fmt.ts";
import type { Outcome, OutcomeOverall, OutcomeRow } from "./types.ts";

const SVG_NS = "http://www.w3.org/2000/svg";

/** Tiles per side, and the grid's total, which the file's `tiles` must equal. */
const SIDE = 10;
const TOTAL = SIDE * SIDE;

/** Geometry in user units: a 40-unit pitch with 2 units of surface showing. */
const PITCH = 40;
const GAP = 2;
const RADIUS = 3;

/** Per-tile delay of the one page-load moment: the last tile starts under a second in. */
const STAGGER_MS = 9;

/** How each outcome reads in a sentence and in the key under the tiles. */
const PLAIN: Record<Outcome, string> = {
  Dropout: "dropped out",
  Enrolled: "were still enrolled",
  Graduate: "graduated",
};

function reducedMotion(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

/** The outcome of each tile, row-major: Dropout first, then Enrolled, then Graduate. */
function tileOrder(tiles: Record<Outcome, number>): Outcome[] {
  const order: Outcome[] = [];
  for (const outcome of OUTCOMES) {
    for (let i = 0; i < tiles[outcome]; i += 1) order.push(outcome);
  }
  return order;
}

function buildGrid(order: Outcome[]): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  const extent = SIDE * PITCH - GAP;
  svg.setAttribute("viewBox", `0 0 ${extent} ${extent}`);
  svg.setAttribute("class", "tile-grid-svg");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  order.forEach((outcome, i) => {
    const rect = document.createElementNS(SVG_NS, "rect");
    const row = Math.floor(i / SIDE);
    const col = i % SIDE;
    rect.setAttribute("x", String(col * PITCH));
    rect.setAttribute("y", String(row * PITCH));
    rect.setAttribute("width", String(PITCH - GAP));
    rect.setAttribute("height", String(PITCH - GAP));
    rect.setAttribute("rx", String(RADIUS));
    rect.setAttribute("data-outcome", outcome);
    // A token rather than a resolved hex, so the theme toggle recolours the
    // tiles with no re-render.
    rect.style.fill = `var(${outcomeToken(outcome)})`;
    rect.style.setProperty("--d", `${i * STAGGER_MS}ms`);
    svg.appendChild(rect);
  });
  return svg;
}

function swatch(outcome: Outcome): HTMLSpanElement {
  const node = el("span", "key-swatch");
  node.style.background = `var(${outcomeToken(outcome)})`;
  node.setAttribute("aria-hidden", "true");
  return node;
}

/** "32 dropped out, 18 were still enrolled and 50 graduated" */
function tileClause(tiles: Record<Outcome, number>): string {
  const parts = OUTCOMES.map((o) => `${tiles[o]} ${PLAIN[o]}`);
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

function rowFor(data: OutcomeOverall, outcome: Outcome): OutcomeRow {
  const row = data.rows.find((r) => r.outcome === outcome);
  if (!row) throw new Error(`outcome_overall.json has no row for ${outcome}`);
  return row;
}

function buildKey(tiles: Record<Outcome, number>): HTMLUListElement {
  const key = el("ul", "tiles-key");
  key.setAttribute("aria-hidden", "true");
  for (const outcome of OUTCOMES) {
    const item = el("li");
    item.appendChild(swatch(outcome));
    item.appendChild(el("span", undefined, `${tiles[outcome]} ${PLAIN[outcome]}`));
    key.appendChild(item);
  }
  return key;
}

function buildTiles(data: OutcomeOverall): DocumentFragment {
  const frag = document.createDocumentFragment();

  const grid = el("div", "tile-grid");
  grid.setAttribute("role", "img");
  grid.setAttribute(
    "aria-label",
    `A ten by ten grid of tiles, one per hundred of the ${int(data.n_total)} students: ${tileClause(data.tiles)}.`,
  );
  const svg = buildGrid(tileOrder(data.tiles));
  grid.appendChild(svg);
  frag.appendChild(grid);

  frag.appendChild(buildKey(data.tiles));

  frag.appendChild(
    detailsTable(
      ["Outcome", "Students", "Share", "Interval"],
      OUTCOMES.map((o) => {
        const r = rowFor(data, o);
        return [o, int(r.n), pct(r.p), ciText(r.lo, r.hi)];
      }),
      "Data behind the tiles",
    ),
  );

  if (!reducedMotion()) {
    svg.classList.add("is-pending");
    // Two frames: one for the pending state to be painted, one for the
    // change that the transition runs from.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => svg.classList.remove("is-pending"));
    });
  }
  return frag;
}

/** The rail's strip: the same split as the tiles, as a stacked bar with a key. */
function fillStrip(strip: HTMLElement, data: OutcomeOverall): void {
  strip.replaceChildren();
  strip.setAttribute("role", "img");
  strip.setAttribute(
    "aria-label",
    `Of every hundred students, ${tileClause(data.tiles)}.`,
  );

  const bar = el("div", "strip-bar");
  for (const outcome of OUTCOMES) {
    const seg = el("span", "strip-seg");
    seg.style.flexGrow = String(rowFor(data, outcome).n);
    seg.style.background = `var(${outcomeToken(outcome)})`;
    bar.appendChild(seg);
  }
  strip.appendChild(bar);

  const key = el("ul", "strip-key");
  for (const outcome of OUTCOMES) {
    const item = el("li");
    item.appendChild(swatch(outcome));
    item.appendChild(
      el("span", undefined, `${outcome} ${pct0(rowFor(data, outcome).p)}`),
    );
    key.appendChild(item);
  }
  strip.appendChild(key);
}

/**
 * Fetch the split, draw the tiles into `#tiles` and the strip into
 * `#outcome-strip`. A failed fetch replaces the tiles with a retry, and the
 * strip stays empty; the story around them is untouched.
 */
export async function renderHero(): Promise<void> {
  const tiles = document.getElementById("tiles");
  const strip = document.getElementById("outcome-strip");
  if (!tiles) return;

  let data: OutcomeOverall;
  try {
    data = await loadJson<OutcomeOverall>("outcome_overall.json");
    const sum = OUTCOMES.reduce((acc, o) => acc + (data.tiles[o] ?? 0), 0);
    if (sum !== TOTAL) {
      throw new Error(
        `outcome_overall.json tiles sum to ${sum}, not ${TOTAL}.`,
      );
    }
  } catch (err) {
    showError(
      tiles,
      err instanceof Error ? err.message : "Could not load the outcome split.",
      () => void renderHero(),
    );
    return;
  }

  tiles.replaceChildren(buildTiles(data));
  if (strip) fillStrip(strip, data);
}
