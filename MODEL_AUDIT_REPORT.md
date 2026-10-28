# FlareGuard model and fusion audit

**Audit date:** 2026-10-25  
**Scope:** Current repository source, locally available tabular files, metric
outputs, and tests. No models were loaded and no training or data preparation
was run.

## Executive conclusion

**Decision B: retain the experimental rule-based heuristic and report learned
multimodal fusion as unavailable.**

The repository contains four separate datasets with incompatible subjects,
observation units, and targets. No common patient key, modality-aligned visit,
or independently defined shared RA-flare outcome exists in the available
interfaces. Building a fusion table by row order, fabricated IDs, or proxy
target substitution would not be scientifically valid.

The Flaredown source supports an exploratory, patient-grouped prediction task
for a **future symptom-worsening proxy**, not a clinician-confirmed RA flare.
Sleep and HRV tasks predict their own dataset labels; the Sri Lankan dataset
uses a cohort-relative month-9 cDAI target. None can serve as the independent
shared label for fusion.

The stored HRV Random Forest score of 1.000 is a held-out **row-level**
result, not an independent participant/session evaluation. The source has no
verified participant or session ID, and the old `datasetId` field is constant.
Exact duplicate-row checks did not explain the perfect score; the underlying
reason for perfect classification remains unverified. It must not be
presented as established generalization or RA-flare performance.

The application source keeps the legacy meta-model disabled. The current
score is produced by a rule-based function and is not a clinically validated
probability.

## Dataset-by-dataset findings

### 1. Flaredown RA symptom dataset

- **Files:** `data/ra.csv` (raw; not scanned during this audit),
  `data/ra_processed_v3.csv`, `data/ra_processed_v3_sensitivity.csv`.
- **Current usable processed sample:** 1,762 checkpoint rows for 71 distinct
  Flaredown user IDs; 6–110 rows per user (median 18). The primary proxy target
  has 418 positive and 1,344 negative rows (23.72%). There are 11 exact
  duplicate processed rows. Grouped splitting by user keeps duplicate rows
  from the same user on one side of a split, but duplicates still affect
  weighting.
- **Target:** `flare_next_3d` is 1 when the mean severity of observed symptom
  days from `t+1` through `t+3` is strictly greater than 1.15 times the
  expanding median severity through `t`. At least one future day is required;
  all three need not be observed. This is an operational symptom-worsening
  definition, not clinician adjudication.
- **Predictors:** past 7-day symptom summary (`past_mean_severity`,
  `past_std_severity`, `past_trend`, `past_last_severity`,
  `past_engagement`), past-only `personal_baseline`, age, and sex. The
  preparation code computes the baseline using observations through `t`, so
  the implementation is past-only. The target is nevertheless mathematically
  defined using that same baseline; it measures whether future reported
  symptoms cross a threshold relative to the patient's history, not a
  separate clinical flare event.
- **Validation:** `1_train_models_v2.py` uses a grouped patient holdout and
  training-only grouped CV for its current code path. `5_evaluate_ra_v3.py`
  and `evaluate_flaredown_fair_baseline.py` use repeated
  `StratifiedGroupKFold` patient splits. These prevent the same patient being
  in train and test within a fold. They do **not** evaluate future calendar
  periods or temporal drift.
- **Current-data reproducibility issue:** the current processed v3 file has
  no `checkpoint_date`, while `1_train_models_v2.py` requires it. Thus the
  current main training script rejects the currently checked processed file;
  the metrics JSON's one-time holdout cannot be reproduced from this file
  using that script without resolving the schema/provenance discrepancy.
  The sensitivity output also lacks `checkpoint_date`, so chronological
  validation cannot be reconstructed from the saved processed files.
- **Disposition:** relevant only as an exploratory model for the defined
  Flaredown symptom-worsening proxy. Never call it a validated 72-hour RA
  flare predictor.

### 2. Sleep Health

- **File:** `data/sleep.csv`.
- **Observed sample:** 374 rows, 374 unique `Person ID` values, one row per
  participant, no exact duplicate rows. Labels: 219 no-disorder rows, 78
  sleep-apnea rows, and 77 insomnia rows.
- **Target:** any nonempty, non-`none` `Sleep Disorder` value versus no
  disorder.
- **Predictors:** 11 other columns after dropping target and `Person ID`:
  demographics, occupation, sleep duration/quality, activity, stress, BMI
  category, blood pressure, heart rate, and daily steps.
- **Validation:** stratified random 80/20 holdout; because each ID appears
  once, rows are also distinct participants. The current trainer fits
  imputation, scaling, and one-hot encoding inside its sklearn pipelines.
- **Limitations/relevance:** this is a sleep-disorder task, not RA or flare
  prediction. It is cross-sectional; no RA alignment or outcome exists.
- **Disposition:** may be reported as a separate exploratory sleep-task
  result with sample size and split clearly stated. It is not evidence for RA
  fusion.

### 3. HRV / stress

- **File:** `data/hrv.csv`.
- **Observed sample:** 369,289 rows, 36 columns, 34 numeric HRV predictors
  after dropping `condition` and `datasetId`; no missing target values or
  missing numeric cells were found in the read-only aggregate scan.
- **Target:** `condition != "no stress"`; source classes are 200,082
  `"no stress"` and 169,207 stress rows (`"interruption"` plus
  `"time pressure"`), giving 45.82% positive.
- **Identifier evidence:** all 369,289 rows have the same `datasetId` value
  (`2`). No verified participant or session identifier is present. Source
  order changes class 219,486 times, so fixed row blocks cannot be treated as
  proven sessions.
- **Duplicate check:** the stream check found 369,289 distinct hashes for
  complete rows and for numeric-feature rows; no exact repeated feature
  vectors or conflicting labels were found by this check. This rules out
  exact duplicates as the observed explanation for the perfect score, but
  does not rule out within-person/session dependence, derived-feature
  leakage, or collection/task artifacts. Descriptive single-feature ranges
  inspected for common HRV measures overlapped across labels; no single
  obvious range threshold was established as the explanation.
- **Current validation:** `1_train_models_v2.py` explicitly removes
  `condition` and `datasetId`, but uses stratified **row-level** train/test
  splitting because no participant/session groups are available. The
  resulting test rows are sampled from the same mixed source. A row-level
  holdout is not an independent-subject or independent-session evaluation.
- **Why the stored Random Forest reached 100%:** the supported conclusion is
  that a Random Forest classified the row-level holdout perfectly under the
  recorded split. The **specific cause is unknown**. No participant/session
  IDs or data-generation provenance exist here to establish independence.
  Exact duplicates and a varying datasetId do not explain it: duplicates were
  not found, and datasetId is constant and excluded. Highly discriminative
  physiological patterns, repeated measurements from shared recording
  sessions, or task/data-construction artifacts remain possible.
- **Prior “leakage correction” caveat:** committed older code grouped by
  `datasetId`, but because that field is constant it is not a valid session
  grouping strategy. It also included a pseudo-session block-by-row-order
  approach; given the observed shuffled condition sequence and absent true
  IDs, those blocks are not verified sessions. The current metrics instead
  explicitly state row-level random holdout. `4_verification_checks.py`
  proposes a 40,000-row sample and feature-cluster tests, but this audit did
  not run it; its proposed cluster scaling is fitted before the cluster split
  and therefore is not a clean held-out preprocessing design.
- **Disposition:** not eligible as a modality for RA-flare fusion. The 1.000
  accuracy must not be presented as reliable stress generalization without
  external validation grouped by known participants/sessions.

### 4. Sri Lankan RA clinical dataset

- **File:** `data/sri_lankan_ra.csv`.
- **Observed sample:** 107 data rows; 103 nonblank unique participant IDs in
  the file. Month-9 cDAI is available for 77 unique participants; 30 rows lack
  that outcome.
- **Target:** `cDAI(9) > 6.0`, where 6.0 is the median computed over the
  entire set of outcome-observed rows. This produces 33 positive and 44
  negative cases. It is a cohort-relative high/low cDAI outcome, not a
  clinician-confirmed flare label.
- **Predictors:** nine baseline variables: age, education, disease duration,
  tender joints, swollen joints, ESR, baseline DAS28, baseline cDAI, and
  baseline pain VAS. Month-4 and month-9 columns are excluded from X.
- **Validation:** current source uses a stratified random holdout. IDs are
  unique for outcome-observed records, so the split separates patients.
  However, the target median is calculated from **all outcome-observed
  records before the split**, allowing held-out test outcomes to influence
  the target threshold. The test set has only 16 rows, and follow-up outcome
  availability is incomplete.
- **Disposition:** exploratory small-cohort cDAI classification at most.
  It is not a flare model, not evidence of a shared label, and the current
  target definition has a test-informed threshold.

## HRV 100% accuracy and leakage review

The checked-in `models/metrics.json` records HRV Random Forest accuracy,
F1, ROC-AUC, PR-AUC, precision, and recall as 1.000 on 73,858 test rows from
a stratified random row split. It records XGBoost at 0.960 accuracy and
0.994 ROC-AUC, and Logistic Regression at 0.686 accuracy. The code and data
support only the statement that the row-level held-out rows were classified
as recorded; they do not support generalization to new people, devices,
sessions, or clinical populations.

Audit findings:

1. The target is taken directly from `condition` and that column is removed
   from X. No explicit target column remains among the selected predictors.
2. `datasetId` is constant and dropped in the current training code. It
   cannot serve as a participant/session grouping variable.
3. Exact duplicate full rows and feature vectors were not observed in the
   aggregate hash scan.
4. The current validation split is random by row. With no participant/session
   IDs, it cannot prevent related observations from crossing the split.
5. Feature/label source provenance is insufficient to rule out systematic
   collection-condition artifacts or hidden dependence.
6. A 100% result is plausible for this dataset/split but its cause and
   external validity remain unverified. It should be withheld from thesis
   claims of independent stress prediction until source provenance and
   subject/session-grouped validation are available.

No HRV model was trained, evaluated, or loaded during this audit.

## Leakage and validation summary

- **Preprocessing:** in the current general trainer, learned imputation,
  scaling, and encoding are placed in sklearn pipelines and fitted on
  training folds. Training-only CV selects the model. This is an improvement
  over the older all-data `LabelEncoder` workflow in `1_train_models.py`.
- **RA Flaredown:** grouped patient splits reduce patient leakage; however,
  no temporal holdout is available in the saved processed file. The
  construction target is an imperfect future symptom proxy; 1–3 observed
  future days define different target availability, motivating the saved
  sensitivity analysis. Exact duplicate rows exist but are grouped by
  patient.
- **Sleep:** participant ID is dropped; current file has one row per unique
  participant, so stratified row split is participant-disjoint. The endpoint
  is not an RA outcome.
- **HRV:** row-level validation is not subject/session-independent. The
  constant dataset ID and pseudo-session block heuristic do not solve this.
- **Sri Lankan RA:** baseline predictors avoid direct inclusion of follow-up
  measures, but median target threshold uses full-cohort month-9 labels before
  splitting; the 16-row test and incomplete follow-up make results unstable.
- **Fusion:** there is no aligned common-cohort table or shared independent
  target. There is no valid basis for training or evaluating a meta-model.

## Current and saved performance results

### `models/metrics.json` (single holdout)

These are historical base-model results and not fusion metrics. The split
metadata records a held-out test set, but these values should not be described
as independently reproduced against the current data/code; the current
Flaredown training path requires `checkpoint_date`, which the current
processed file lacks.

| Dataset/target | Algorithm | Test rows | Accuracy | ROC-AUC | PR-AUC | Precision / recall |
|---|---|---:|---:|---:|---:|---:|
| Flaredown symptom-worsening proxy | Logistic Regression | 281 | 0.673 | 0.710 | 0.448 | 0.404 / 0.657 |
| Flaredown symptom-worsening proxy | Random Forest | 281 | 0.762 | 0.678 | 0.446 | 0.579 / 0.157 |
| Flaredown symptom-worsening proxy | XGBoost | 281 | 0.683 | 0.674 | 0.432 | 0.373 / 0.400 |
| Sleep disorder vs none | Logistic Regression | 75 | 0.893 | 0.947 | 0.923 | 0.829 / 0.935 |
| Sleep disorder vs none | Random Forest | 75 | 0.907 | 0.952 | 0.945 | 0.853 / 0.935 |
| Sleep disorder vs none | XGBoost | 75 | 0.933 | 0.954 | 0.954 | 0.906 / 0.935 |
| HRV condition stress proxy | Logistic Regression | 73,858 | 0.686 | 0.748 | 0.721 | 0.679 / 0.597 |
| HRV condition stress proxy | Random Forest | 73,858 | **1.000** | **1.000** | **1.000** | **1.000 / 1.000** |
| HRV condition stress proxy | XGBoost | 73,858 | 0.960 | 0.994 | 0.993 | 0.954 / 0.958 |
| Month-9 cDAI above cohort median | Logistic Regression | 16 | 0.500 | 0.635 | 0.577 | 0.000 / 0.000 |
| Month-9 cDAI above cohort median | Random Forest | 16 | 0.688 | 0.587 | 0.567 | 0.667 / 0.571 |
| Month-9 cDAI above cohort median | XGBoost | 16 | 0.500 | 0.476 | 0.437 | 0.444 / 0.571 |

### Repeated grouped Flaredown proxy evaluations

The saved `outputs/ra_v3_fair_baseline_results.json` and
`outputs/ra_v3_results.json` describe 6 repeats of 5-fold patient-grouped
evaluation on 1,762 rows and 71 patients. The recorded Logistic Regression
means are ROC-AUC 0.786 (SD 0.034) and PR-AUC 0.542 (SD 0.066); the naive
past-mean/personal-baseline ratio is ROC-AUC 0.725 and PR-AUC 0.477 in the
fair-baseline output. The baseline omits rows with zero/invalid baselines, so
its evaluation population differs. Repeated fold standard deviations are not
independent-sample confidence intervals, and selecting a winner based on the
same repeated folds can introduce selection optimism.

These may be reported only as **exploratory grouped estimates for the
operational symptom-worsening proxy**, with the target, cohort size, grouped
protocol, baseline comparison, no temporal validation, and non-clinical
label limitations stated. They do not establish clinical flare performance.
Sensitivity CSV results for requiring 1, 2, or 3 observed future days are
likewise proxy-task sensitivity results, not three clinical outcome labels.

### Thesis reporting recommendation

- **May report cautiously:** separate Sleep Health target metrics as
  dataset-specific sleep-disorder results; grouped Flaredown metrics as
  exploratory symptom-worsening proxy results, preferably with the repeated
  grouped evaluation and naive baseline rather than the unverified single
  holdout.
- **Report only as an explicitly limited exploratory result, or omit
  performance claims:** Sri Lankan RA cDAI classification due to the
  full-cohort median threshold, 77 labeled patients, and 16 held-out rows.
- **Do not report as validated or independent performance:** HRV Random
  Forest 100% (or the other row-level HRV metrics) without subject/session
  validation and source provenance.
- **Do not report any fusion metrics:** none were generated from aligned
  independent fusion examples. The legacy `meta_model.pkl` is not valid
  evidence.
- All base-model metrics are separate task metrics; they are not fusion
  results and do not validate the live heuristic or clinical use.

## Fusion feasibility decision

**No valid fusion training dataset is present.** Flaredown has user IDs and
an unlabeled-as-clinical symptom proxy. Sleep Health uses its own `Person ID`
and sleep-disorder label. HRV provides a constant `datasetId`, no verified
participant/session ID, and a stress-condition label. The Sri Lankan
clinical cohort has its own `Index No` and month-9 cDAI, with no mapping to
the other datasets or independent flare adjudication. These identifiers
cannot be linked across sources based on the available files.

There is consequently no scientifically justified example-level join,
pairing strategy, or independent common target. A supervised meta-classifier
cannot be evaluated independently with these data. `2_fusion.py` sets
`ModelBundle.meta` to `None`, keeps `fusion_available` false, excludes
`meta_model.pkl` from `_MODEL_SPECS`, and implements `compute_risk` as a
rule-based score that does not call base-model predictions.

## Recommended next actions

1. Keep the live experimental heuristic separate from model outputs; state
   learned fusion is unavailable and do not report fusion performance.
2. If thesis space permits, report the Flaredown proxy as an exploratory
   longitudinal task and explain it is not clinician-adjudicated RA flare.
   State that no temporal holdout was possible with the saved processed file.
3. Resolve the Flaredown schema/provenance gap (`checkpoint_date` required by
   the trainer but absent from the current processed file) before claiming
   reproducibility. Do not regenerate the dataset without approval.
4. Treat HRV 100% as an unresolved row-level anomaly. Obtain verified subject,
   recording-session, and task/condition provenance from the dataset source;
   then evaluate on strictly held-out participants/sessions with all
   learned preprocessing and feature selection fitted within training folds.
5. For Sri Lankan RA, define a prespecified clinically meaningful target or
   a training-only threshold, keep test labels out of target construction,
   and obtain a larger independent cohort before making predictive claims.
6. A future fusion study requires a prospective or otherwise genuinely
   aligned cohort with synchronized modality measurements, a prespecified
   independent clinician-adjudicated RA flare onset/outcome and time window,
   enough patients and events, patient/time-separated partitions, and
   comparison against modality-specific and simple clinical baselines.
   Calibration and external validation would still be required; even that
   would not by itself establish clinical utility.

## Files inspected and checks run

Inspected source: `1_train_models.py`, `1_train_models_v2.py`,
`2_fusion.py`, `3_patient_console.py`, `4_verification_checks.py`,
`5_evaluate_ra_v3.py`, `prepare_flaredown_v2.py`,
`prepare_flaredown_sensitivity.py`, `evaluate_flaredown_fair_baseline.py`,
`evaluate_flaredown_sensitivity.py`, `api.py`, `database.py`, README and
deployment configuration; inspected `models/metrics.json`, saved evaluation
summaries, and test files. Dataset inspection was limited to table schemas,
counts, and aggregate duplicate/label checks; no patient IDs or identifying
records were printed.

Tests run:

- `python -m unittest discover -s tests -p "test_fusion_safety.py" -v` —
  **14 passed**.
- `python -m unittest discover -s tests -p "test_prepare_flaredown_v2.py" -v`
  — **15 passed**. These are fixture/unit tests; this did not run full
  preparation.

No training, model/artifact deserialization, full raw Flaredown preparation,
artifact overwrite, staging, commit, or deployment was performed.
