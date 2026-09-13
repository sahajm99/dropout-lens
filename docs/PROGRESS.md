# Progress log

One entry per milestone: what was verified and how.

## 2026-09-12: Milestone 0, setup

- `gh auth switch --user sahajm99`; repo-local git identity `sahajm99 <64627746+sahajm99@users.noreply.github.com>`, LF endings.
- Copied `dataset.csv` (SHA-256 `ff327a7e...0f4e`) to `data/raw/` with `SOURCE.md`; the Fall 2023 and Spring 2024 course copies are byte-identical (`cmp`).
- Two scouting subagents (read-only on the course folders) wrote `docs/SOURCES.md` (the questions, charts, tests and numbers the two course projects claimed) and `docs/DATA_AUDIT.md` (shape, value counts, code-mapping verification, anomalies). Findings that changed the design: 34 features not 36; attendance is a function of course; tuition and debtor status are near-outcome flags; the 180 zero-enrolment rows are one course's record gap.
- Palette validated with the dataviz skill's validator on both chosen surfaces (`#f3f5f8`, `#0f1622`), all-pairs, three outcome slots: all checks pass; light-mode orange and aqua need direct labels (relief rule).
- `docs/DESIGN.md`, `docs/DECISIONS.md` (D1 to D8, S1 to S11, W1 to W10, P1 to P5, rulings R1 to R7) and the plan `docs/superpowers/plans/2026-09-12-dropout-lens.md`.
- Scaffold committed (`34f85c9`): `pyproject.toml` with `uv.lock` (pandas 2.3.3, numpy 2.5.3, statsmodels 0.14.6, scikit-learn 1.9.1, scipy 1.18.1), `site/` (Vite 8, TypeScript 6, vitest 4, plotly.js-cartesian-dist-min 4, Literata and Bricolage Grotesque from fontsource), `.github/workflows/ci.yml`, `scripts/check_stale.py` (tolerance diff).
- Public repo `sahajm99/dropout-lens` created and pushed; Pages set to `build_type=workflow` (`gh api`), URL `https://sahajm99.github.io/dropout-lens/`.
