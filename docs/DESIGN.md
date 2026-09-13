# Dropout Lens: design

A live, static analysis of one Portuguese polytechnic's enrolment records
(Realinho, Vieira Martins, Machado and Baptista, UCI "Predict Students' Dropout
and Academic Success", CC BY 4.0: 4,424 students enrolled between 2008 and 2019,
34 features in this copy (the UCI page lists 36; the two admission-grade columns are absent here), outcome Dropout / Enrolled / Graduate measured at the normal end of
the course). It rebuilds two UNT course projects on the same file: the Fall 2023
Fundamentals of AI project (EDA and visualisation) and the Spring 2024 Empirical
Analysis project (chi-square, t-tests, ANOVA). Both are documented in
`docs/SOURCES.md`; the data audit is `docs/DATA_AUDIT.md`.

## The problem

The portfolio card for this work says "10,000+ student records" and "85% accuracy
on holdout set". The file has 4,424 rows, and neither course project measured a
held-out accuracy under a fixed split. The rebuild measures everything, publishes
the measured number whatever it is, and corrects the card.

## Audience and job

A hiring manager or engineer with two minutes. They should be able to see who is in
the data, what differs at enrolment, what the tests say with effect sizes rather than
stars, what a model can and cannot predict and why the two feature sets are not
comparable, and put a profile through the enrolment model in the browser. Every
number on the page is read from a JSON file the pipeline wrote.

## Questions the page answers, in order

1. Who is in the data and how does it end? Hero: the three-way split.
2. What differs at enrolment? Outcome shares with Wilson intervals by gender,
   scholarship, tuition status, debtor, age band, application mode, course,
   attendance and displacement; then the first-semester approval band, which is
   marked as a post-enrolment variable.
3. What do the tests say? Chi-square with Cramér's V for categorical predictors,
   ANOVA and Kruskal-Wallis with eta squared for age and semester grades, raw and
   Holm-adjusted p-values side by side.
4. What can a model predict, and when? Binary dropout and three-class outcome,
   four models, two feature sets ("at enrolment" and "after first semester"), test
   metrics with bootstrap intervals, calibration.
5. What drives the model? Odds ratios from the pinned-reference logistic models and
   permutation importance on the test set.
6. Where is it unequal? Per-group false negative and false positive rates of the
   best model, with intervals.
7. Put a profile through the enrolment model. Eight inputs, a probability with an
   interval, the held-at-reference fields stated.
8. What this cannot tell you.

## Visual identity

Distinct from CardioLens (cream, Source Serif and Sans, top nav, stat tiles) and
Broadsheet (warm pink, IBM Plex, newspaper rules).

- Subject cue: Portuguese tilework. The hero is a 10 by 10 grid of tiles, one per
  hundred students, coloured by outcome, drawn as SVG from the data. The same
  three-way split lives as a slim vertical strip in the left rail on wide screens
  and as a thin horizontal bar under the header on narrow ones, so the split is
  always in view. That grid is the one loud element; everything else is quiet.
- Colour: light surface `#f3f5f8` (cool plaster), second surface `#e6eaf0`, ink
  `#0f1a2b`, secondary ink `#45506a`, tertiary `#7c869b`, grid `#d5dbe5`, accent
  cobalt `#1d4ea3`. Dark surface `#0f1622`, second `#18222f`, ink `#e9edf3`,
  secondary `#b4bccb`, tertiary `#7f8a9d`, grid `#26303f`, accent `#6ea3f0`.
  Outcome colours are the dataviz reference palette's first three categorical
  slots, assigned once and never re-ordered: Graduate blue `#2a78d6` (dark
  `#3987e5`), Dropout orange `#eb6834` (dark `#d95926`), Enrolled aqua `#1baf7a`
  (dark `#199e70`). Validated with the skill's validator on both surfaces, all-pairs:
  every check passes; light-mode orange and aqua sit under 3:1 on the surface, so
  every figure carries direct labels and a table (the relief rule). Sequential
  blue ramp for magnitude; no diverging scale is needed.
- Type: Literata Variable for prose (a serif made for long reading on screens,
  optical size axis on) and Bricolage Grotesque Variable for the headline,
  headings, rail, controls, tiles and chart text. The hero headline is Bricolage
  at width 87, weight 600, tight leading. Sentence case everywhere; no all-caps
  labels; no eyebrows; no numbered section markers.
- Layout: at 960px and up, a two-column grid: a 250px sticky left rail (site
  name, the eight questions as the section list, the outcome strip, the theme
  toggle) and the story column, prose at 66ch and figures up to 860px, left
  aligned. Under 960px the rail becomes a top bar with a horizontally scrolling
  section list. Works at 400px.
- Motion: one page-load moment, the hero tiles filling in outcome by outcome;
  disabled under reduced motion. No entrance animation on anything else.
- Structure encodes meaning: any figure or metric that uses the after-first-semester
  feature set carries a left rule in the accent colour and the word "leaks" in its
  context line, so the reader always knows which feature set they are looking at.
- Copy: every heading is a question; every chart title is the answer as a sentence
  with its number filled from data. The estimator says "an association in one
  Portuguese institution's 2008 to 2019 records, not a prediction about a person and
  not a decision tool". The phrase "risk score" never appears.

## Pipeline

`python -m pipeline` (Python 3.12, uv; pandas, numpy, statsmodels, scikit-learn,
scipy) reads `data/raw/dataset.csv` and writes JSON under 200 KB each to
`site/public/data/`, `claims.json` and `meta.json` last. Modules: `load`
(labels and derived columns), `stats` (Wilson, Cramér's V, eta squared, Holm),
`descriptive`, `hypothesis`, `features`, `metrics`, `models`, `fairness`,
`estimator`, `claims`. Fixed seed 20260912 everywhere; single-threaded model
fitting so the committed JSON reproduces on CI.

## Site

Vite + TypeScript + `plotly.js-cartesian-dist-min`, no framework, no CDN, fonts
self-hosted, light and dark, responsive to 400px. Same chrome shape as CardioLens
(theme tokens, lazy figures, `data-stat` spans filled from `claims.json`, title
templates, a details table under every figure) with the identity above. Twelve
figures: the tile grid, outcome shares by group (stacked) and dropout share by group
with intervals (both with a picker), the first-semester approval band, effect sizes,
age and grades by outcome, binary model metrics, three-class model metrics,
calibration, the odds-ratio forest with a feature-set toggle, permutation importance,
the fairness slice, and the estimator panel.

## Definition of done

Public repo `sahajm99/dropout-lens` (MIT, README with a green CI badge), live at
<https://sahajm99.github.io/dropout-lens/>, the data committed with `SOURCE.md`,
CI running the pipeline, a staleness check, pytest, typecheck, vitest parity, build
and deploy; the portfolio entry corrected in place and Vercel READY.
