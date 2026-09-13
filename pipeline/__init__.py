"""Dropout Lens pipeline: reads data/raw/dataset.csv once and writes site/public/data/*.json.

OMP_NUM_THREADS is pinned here, before any module can import scikit-learn, so that
model fitting is single-threaded and the committed JSON reproduces on CI.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
