import { loadJson } from "../data.ts";
import { evaluate, type EstimatorFile } from "../estimator-core.ts";
import { el, showError } from "../figure.ts";
import { lowerLead, pct } from "../fmt.ts";

/**
 * What comes out of this panel, verbatim, every time: an association in one
 * institution's records over one span of years, not a forecast about anyone
 * and not something to act on. It never labels the output as a score for a person.
 */
const CAVEAT =
  "What this panel shows is an association in one Portuguese institution's 2008 to 2019 records, not a prediction about a person and not a decision tool. Nothing here should inform an admissions, funding or advising decision about any one student.";

/** The two marks on the scale, in the order they are drawn and keyed. */
const MARKS = [
  { key: "profile", label: "This profile" },
  { key: "all", label: "Training-set average" },
] as const;

/**
 * `held_at_reference` read as a sentence, e.g. "Held at their reference
 * level: fees up to date, not a debtor, Portuguese, no special needs." Built
 * from the file so a pipeline change to the held set changes this sentence
 * along with it, rather than leaving stale wording behind.
 */
function heldSentence(est: EstimatorFile): string {
  const phrases = est.held_at_reference.map((item) => lowerLead(item.level));
  return `Held at their reference level: ${phrases.join(", ")}.`;
}

function position(value: number, max: number): string {
  const share = max > 0 ? value / max : 0;
  return `${(Math.min(1, Math.max(0, share)) * 100).toFixed(2)}%`;
}

/** The scale: a CSS track with a mark per number in the sentence above it. */
function buildScale(): {
  node: HTMLDivElement;
  range: HTMLDivElement;
  marks: Map<string, HTMLSpanElement>;
  max: HTMLSpanElement;
} {
  const node = el("div", "est-scale");
  node.setAttribute("aria-hidden", "true");

  const track = el("div", "est-track");
  const range = el("div", "est-range");
  track.appendChild(range);
  const marks = new Map<string, HTMLSpanElement>();
  for (const mark of MARKS) {
    const span = el("span", `est-mark est-mark-${mark.key}`);
    marks.set(mark.key, span);
    track.appendChild(span);
  }
  node.appendChild(track);

  const ends = el("div", "est-ends");
  ends.appendChild(el("span", undefined, pct(0, 0)));
  const max = el("span");
  ends.appendChild(max);
  node.appendChild(ends);

  const key = el("ul", "est-key");
  for (const mark of MARKS) {
    const item = el("li");
    item.appendChild(el("span", `est-swatch est-mark-${mark.key}`));
    item.appendChild(document.createTextNode(mark.label));
    key.appendChild(item);
  }
  node.appendChild(key);

  return { node, range, marks, max };
}

export async function render(container: HTMLElement): Promise<void> {
  let est: EstimatorFile;
  try {
    est = await loadJson<EstimatorFile>("estimator.json");
  } catch (err) {
    showError(
      container,
      err instanceof Error ? err.message : "Could not load the model inputs.",
      () => void render(container),
    );
    return;
  }

  container.replaceChildren();
  container.appendChild(
    el("h3", "fig-title", "Put a profile through the enrolment model"),
  );
  container.appendChild(
    el(
      "p",
      "fig-sub",
      "Fitted probabilities from the at-enrolment logistic model; every " +
        "answer it takes is on this list, and the rest are held at their " +
        "reference level.",
    ),
  );

  // The form is built from `inputs` only: `coefficients` carries every level
  // the model fitted, including levels of covariates no reader is asked
  // about because they are held at their reference level instead.
  const chosen: Record<string, string> = {};
  const form = el("form", "est-form");
  form.addEventListener("submit", (e) => e.preventDefault());
  for (const input of est.inputs) {
    chosen[input.variable] = input.default;
    const field = el("div", "est-field");
    const select = el("select");
    select.id = `est-${input.variable}`;
    for (const level of input.levels) {
      const opt = el("option", undefined, level);
      opt.value = level;
      opt.selected = level === input.default;
      select.appendChild(opt);
    }
    select.addEventListener("change", () => {
      chosen[input.variable] = select.value;
      update();
    });
    const label = el("label", undefined, input.label);
    label.htmlFor = select.id;
    field.append(label, select);
    form.appendChild(field);
  }
  container.appendChild(form);

  const out = el("div", "est-out");
  const value = el("p", "est-value");
  const sentence = el("p", "est-sentence");
  out.append(value, sentence);
  const scale = buildScale();
  const held = el("p", "est-held", heldSentence(est));
  const caveat = el("p", "est-caveat", CAVEAT);

  function update(): void {
    let p: number;
    let lo: number;
    let hi: number;
    try {
      ({ p, lo, hi } = evaluate(est, chosen));
    } catch (err) {
      showError(
        container,
        err instanceof Error
          ? err.message
          : "The model file is missing something this panel needs.",
        () => void render(container),
      );
      return;
    }

    // Room above the highest mark, but never past 100%: this is a share of
    // students, and the reader should be able to see the average too.
    const max = Math.min(1, Math.max(0.3, hi * 1.2, est.base_rate * 1.2));

    value.textContent = pct(p);
    sentence.textContent =
      `Students with this profile left at ${pct(p)} ` +
      `(95% interval ${pct(lo)} to ${pct(hi)}). ` +
      `The training-set average was ${pct(est.base_rate)}.`;

    scale.range.style.setProperty("left", position(lo, max));
    scale.range.style.setProperty(
      "width",
      `calc(${position(hi, max)} - ${position(lo, max)})`,
    );
    scale.marks.get("profile")?.style.setProperty("left", position(p, max));
    scale.marks
      .get("all")
      ?.style.setProperty("left", position(est.base_rate, max));
    scale.max.textContent = pct(max, 0);
  }

  // The first value is written before the live region joins the document, so
  // a screen reader reads it as part of the panel rather than announcing it
  // as a change the reader did not make.
  update();
  out.setAttribute("aria-live", "polite");
  container.append(out, scale.node, held, caveat);
}
