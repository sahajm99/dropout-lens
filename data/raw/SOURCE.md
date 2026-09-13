# Data source

**File:** `dataset.csv` (470,858 bytes, 4,424 rows, 34 features plus `Target`, UTF-8 with BOM)
**SHA-256:** `ff327a7ed348140d42dd682df49e36331e2ca91ca41b8fb9ad911a4f5fba0f4e`

**Origin:** UCI Machine Learning Repository, *Predict Students' Dropout and Academic
Success*, <https://doi.org/10.24432/C5MC89>, donated 2021-12-12. The data were
collected at the Instituto Politécnico de Portalegre, Portugal, for students
enrolled in undergraduate degrees between the 2008/09 and 2018/19 academic years,
and were assembled and published by the authors of the paper below.

**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
<https://creativecommons.org/licenses/by/4.0/>. This copy is committed unmodified so
the pipeline can be rerun against it; the attribution above and the citation below
are the licence's condition.

## Citation

Realinho, V.; Machado, J.; Baptista, L.; Martins, M. V. Predicting Student Dropout
and Academic Success. *Data* 2022, 7(11), 146. <https://doi.org/10.3390/data7110146>

Martins, M. V.; Tolledo, D.; Machado, J.; Baptista, L. M. T.; Realinho, V. Early
Prediction of Student's Performance in Higher Education: A Case Study. In *Trends and
Applications in Information Systems and Technologies*, WorldCIST 2021, Advances in
Intelligent Systems and Computing 1365, Springer, 2021.
<https://doi.org/10.1007/978-3-030-72657-7_16>

## Columns

Thirty-four features known at enrolment or at the end of the first and second
semesters, and the outcome `Target` with values `Dropout`, `Enrolled` and `Graduate`,
recorded at the normal end of the course. Categorical features are integer codes; this
file uses the sequential codes of Table 1 in Realinho et al. (2022), not the raw
institutional codes shown on the UCI page. The decoding used by the pipeline is
`pipeline/constants.py`, and `docs/DATA_AUDIT.md` records the empirical checks that
support it (evening courses carry attendance 0; every applicant under the over-23
route is at least 23; the international application modes carry the International
flag).

## Known properties that shape the analysis

- One institution, one country; nothing here generalises on its own.
- The outcome is measured at the normal course length, so `Enrolled` means "not
  finished yet", which is neither success nor failure.
- Semester results are precursors of the outcome; models that use them are reported
  separately and marked as leaking.
- No student identifier; no rows are exact duplicates.
- Nationality is over 97% Portuguese; parents' qualification and occupation carry
  dozens of thin coded levels and are not used.

## Never modified

This file is committed as downloaded. The pipeline reads it and writes aggregates to
`site/public/data/`; nothing writes back here.
