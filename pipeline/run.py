"""Pipeline entry point: loads the data once and runs every registered writer.

The run is idempotent: stale `*.json` are deleted from `OUT_DIR` first, so a file
that a writer stops producing cannot linger and be served by the site. Writers
run in dependency order -- the descriptive pair, the hypothesis tests, the models
(which fit once and are memoised for the fairness and estimator writers), and
`write_claims_and_meta` last because it reads the files the others just wrote.
"""

from __future__ import annotations

from pipeline.claims import write_claims_and_meta
from pipeline.constants import OUT_DIR
from pipeline.descriptive import write_outcome_by_group, write_outcome_overall
from pipeline.estimator import write_estimator
from pipeline.fairness import write_fairness
from pipeline.hypothesis import write_hypothesis_tests
from pipeline.load import load_data
from pipeline.models import write_models

WRITERS = [
    write_outcome_overall,
    write_outcome_by_group,
    write_hypothesis_tests,
    write_models,
    write_fairness,
    write_estimator,
    # Reads every file above back from disk, so it must stay last.
    write_claims_and_meta,
]


def clean_out_dir() -> None:
    """Remove stale JSON so the run's output set is exactly what the writers emit."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("*.json"):
        path.unlink()


def run() -> list:
    df = load_data()
    clean_out_dir()
    paths = []
    for writer in WRITERS:
        # A writer returns either a single Path or a list of Paths.
        written = writer(df)
        for path in written if isinstance(written, list) else [written]:
            print(f"{path.name}  {path.stat().st_size}")
            paths.append(path)
    return paths
