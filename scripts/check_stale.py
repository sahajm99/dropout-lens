"""Fail if the committed JSON under site/public/data/ differs from a fresh pipeline run.

Run after `python -m pipeline` has overwritten the files. Each file is compared with
its committed version (`git show HEAD:<path>`): `generated_at` is ignored, numbers
are compared with a tolerance (1e-6 by default; 5e-3 for the model files, whose
tree fits can move by one test row between platforms), everything else exactly.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path("site/public/data")
MODEL_FILES = {
    "model_metrics.json",
    "permutation_importance.json",
    "fairness.json",
    "calibration.json",
    "odds_ratios.json",
}
LOOSE_TOL = 5e-3
TIGHT_TOL = 1e-6
# The three-class logit has one quasi-separated level (docs/DECISIONS.md R9), so
# its optimiser stops at a platform-dependent point: Newton on one BLAS, BFGS on
# another. The non-separated coefficients agree to about 1e-3 and the wide
# interval bounds to about 2e-2 across those paths, so that block gets its own
# tolerance. A confusion-matrix cell is a count of test rows and may move by
# one when a probability sits on the threshold.
MULTICLASS_LOGIT_TOL = 5e-2
CONFUSION_TOL = 2.0


def tolerance(file_tol: float, where: str) -> float:
    if ".multiclass_enrolment." in where:
        return MULTICLASS_LOGIT_TOL
    if ".confusion[" in where:
        return CONFUSION_TOL
    return file_tol


def committed(path: Path) -> dict | None:
    rel = path.as_posix()
    res = subprocess.run(
        ["git", "show", f"HEAD:{rel}"], capture_output=True, text=True, encoding="utf-8"
    )
    if res.returncode != 0:
        return None
    return json.loads(res.stdout)


def diffs(a, b, tol: float, where: str = "$") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        keys = set(a) | set(b)
        for k in sorted(keys):
            if k == "generated_at":
                continue
            if k not in a or k not in b:
                out.append(f"{where}.{k}: present on one side only")
                continue
            out.extend(diffs(a[k], b[k], tol, f"{where}.{k}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diffs(x, y, tol, f"{where}[{i}]"))
        return out
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a == b else [f"{where}: {a!r} vs {b!r}"]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        here = tolerance(tol, where)
        rel = 0.01 if ".multiclass_enrolment." in where else 0.0
        if math.isclose(a, b, rel_tol=rel, abs_tol=here):
            return []
        return [f"{where}: {a!r} vs {b!r} (tol {here})"]
    return [] if a == b else [f"{where}: {a!r} vs {b!r}"]


def main() -> int:
    fresh_files = sorted(OUT_DIR.glob("*.json"))
    if not fresh_files:
        print(f"::error::no JSON under {OUT_DIR}; run python -m pipeline first")
        return 1
    failed = False
    for path in fresh_files:
        head = committed(path)
        if head is None:
            print(f"::error::{path.as_posix()} is not committed; run python -m pipeline and commit it")
            failed = True
            continue
        fresh = json.loads(path.read_text(encoding="utf-8"))
        tol = LOOSE_TOL if path.name in MODEL_FILES else TIGHT_TOL
        found = diffs(head, fresh, tol)
        if found:
            failed = True
            print(f"::error::{path.as_posix()} is stale ({len(found)} differences); run python -m pipeline and commit")
            for line in found[:20]:
                print(f"  {line}")
        else:
            print(f"ok  {path.name}")
    # A committed file the pipeline no longer writes would be served stale.
    res = subprocess.run(
        ["git", "ls-tree", "--name-only", "HEAD", f"{OUT_DIR.as_posix()}/"],
        capture_output=True, text=True, encoding="utf-8",
    )
    fresh_names = {p.name for p in fresh_files}
    for line in res.stdout.splitlines():
        name = Path(line).name
        if name.endswith(".json") and name not in fresh_names:
            print(f"::error::{line} is committed but the pipeline no longer writes it")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
