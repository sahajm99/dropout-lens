import "@fontsource-variable/literata/opsz.css";
import "@fontsource-variable/literata/opsz-italic.css";
import "@fontsource-variable/bricolage-grotesque/wdth.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/layout.css";
import "./styles/figure.css";
import "./styles/panel.css";

import { loadJson } from "./data.ts";
import { el } from "./figure.ts";
import { renderHero } from "./hero.ts";
import { whenVisible } from "./lazy.ts";
import { initNav } from "./nav.ts";
import { fillStats } from "./stats-fill.ts";
import { initTheme, toggleTheme } from "./theme.ts";
import type { Claims } from "./types.ts";

interface ChartModule {
  render: (container: HTMLElement) => Promise<void>;
}

/** One dynamic import per chart, so Plotly loads only when a figure needs it. */
const CHARTS: Record<string, () => Promise<ChartModule>> = {
  "outcome-by-group": () => import("./charts/outcomeByGroup.ts"),
  "dropout-by-group": () => import("./charts/dropoutByGroup.ts"),
  "approval-band": () => import("./charts/approvalBand.ts"),
  "effect-sizes": () => import("./charts/effectSizes.ts"),
  distributions: () => import("./charts/distributions.ts"),
  "metrics-binary": () => import("./charts/metricsBinary.ts"),
  "metrics-multiclass": () => import("./charts/metricsMulticlass.ts"),
  calibration: () => import("./charts/calibration.ts"),
  forest: () => import("./charts/forest.ts"),
  importance: () => import("./charts/importance.ts"),
  fairness: () => import("./charts/fairness.ts"),
  estimator: () => import("./charts/estimator.ts"),
};

/** The first figure after the hero starts at once; everything below it waits. */
const EAGER = new Set(["outcome-by-group"]);

const NOTICE_ID = "claims-notice";

/**
 * Every number quoted in the prose comes from claims.json, so a failure to
 * load it leaves placeholders in the middle of sentences. That is worth a
 * visible notice at the top of the story, not only a console line.
 */
function showClaimsNotice(): void {
  const main = document.querySelector("main");
  if (!main || document.getElementById(NOTICE_ID)) return;
  const box = el("div", "fig-error page-notice");
  box.id = NOTICE_ID;
  box.setAttribute("role", "status");
  // Mounted empty, then filled, so the message is announced as a change.
  main.prepend(box);
  box.appendChild(
    el(
      "p",
      undefined,
      "The numbers quoted in the text below could not be loaded, so they are showing as placeholders. The charts fetch their own data and are not affected.",
    ),
  );
  const btn = el("button", "fig-retry", "Try again");
  btn.type = "button";
  btn.addEventListener("click", fillNumbers);
  box.appendChild(btn);
}

function fillNumbers(): void {
  loadJson<Claims>("claims.json")
    .then((claims) => {
      fillStats(claims);
      document.getElementById(NOTICE_ID)?.remove();
    })
    .catch((err: unknown) => {
      console.error("Could not fill the numbers in the prose", err);
      showClaimsNotice();
    });
}

initTheme();
initNav();
document
  .getElementById("theme-toggle")
  ?.addEventListener("click", toggleTheme);

fillNumbers();
void renderHero();

for (const node of document.querySelectorAll<HTMLElement>("[data-chart]")) {
  const name = node.dataset.chart ?? "";
  const load = CHARTS[name];
  if (!load) {
    console.warn(`main: no chart module registered for "${name}"`);
    continue;
  }
  const start = () => {
    load()
      .then((mod) => mod.render(node))
      .catch((err: unknown) => {
        console.error(`main: chart "${name}" failed`, err);
      });
  };
  if (EAGER.has(name)) start();
  else whenVisible(node, start);
}
