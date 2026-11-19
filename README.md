# BigData2026-Assignment2

**Updates are marked with ⭐.**

In this assignment you predict what GitHub users will do in the coming week. For each test
user, you predict which of 6 activities (opening a pull request, pushing code, and so on) the
user will perform during the test week.

This repository contains:

```
assignment2/
+-- README.md                  # download, build the tables, submit
+-- SCHEMA.md                  # every column, explained
+-- scripts/
|   +-- preprocess.py          # builds the 8 tables from GH Archive
|   +-- evaluate.py            # checks the format, computes the score
|   +-- cohort.txt             # contains 9,000 anonymized user IDs (read by preprocess.py)
+-- submission/
    +-- test_input.csv         # official test users
    +-- sample_submission.csv  # example output in the correct format
```

## ⭐ 1. Download the GH Archive files

Download the hourly files of the time period you want from [GH Archive](https://www.gharchive.org)
into your own folder `RAW_DATA_FOLDER`. GH Archive provides one file per hour, named by UTC date
and hour.

For example, based on the command instructions from GH Archive, you can download all the data in
Oct. 2026 using the **GH Archive API**:

```bash
wget -P RAW_DATA_FOLDER https://data.gharchive.org/2026-10-{01..31}-{0..23}.json.gz
```

For more download options, see the GH Archive website.

## ⭐ 2. Build the tables

```bash
python scripts/preprocess.py --raw RAW_DATA_FOLDER --out YOUR_OUTPUT_FOLDER
```

- `RAW_DATA_FOLDER`: your own folder used in Section 1.
- `YOUR_OUTPUT_FOLDER`: your own folder for the tables.

You are required to run our provided preprocessing script before using the data for your model.
The preprocessing script generates the 8 required tables and includes anonymization.

## 3. The tables

| Table | Each row is | Row time |
|---|---|---|
| `Users` | a GitHub user | `first_seen_at` |
| `Repositories` | a repository | `first_seen_at` |
| `PullRequests` | a pull request | `first_seen_at` |
| `Issues` | an issue (the conversation of a pull request is also stored here, with `is_pull_request = true`) | `first_seen_at` |
| `IssueComments` | a comment in an issue or pull request conversation | `created_at` |
| `PRReviews` | a review of a pull request | `first_seen_at` |
| `PRReviewComments` | a comment on specific lines of a pull request's code | `created_at` |
| `Pushes` | a push of commits to a branch or tag | `created_at` |

The **row time** is the time at which a row appears in the data. For a prediction at time t, use
only rows whose row time is before t. Do not filter pull requests, issues or reviews by
`created_at` or `submitted_at`: they can enter the data long after they were created.

**Which users are included.** The tables contain all the activity of 9,000 fixed users (the
*cohort*, marked with `Users.is_cohort = true`), plus the users related to their activity.

**Times** are in UTC (no time zone is stored). Taiwan time is UTC+8.

## 4. Test users and output

⭐ **Test week:** 2026/11/22 00:00 – 11/29 00:00 (Taiwan time)

⭐ **Missing data:** Some hours are missing or incomplete in GH Archive itself. If GH Archive is
missing the data of an activity during the test week, that activity is not scored.

`submission/test_input.csv` lists the test users. Its `UserID` column is `Users.github_user_id`.

Your output is a CSV file with one row per test user, exactly this header, and 1 (the user will
do it) or 0 (the user will not) in each activity column, with no empty cells:

```
UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push
```

**Example output file:** `submission/sample_submission.csv` (random predictions in the correct
format).

| Column | 1 means that during the test week the user ... |
|---|---|
| `open_pr` | opens a pull request |
| `open_issue` | opens an issue (not a pull request) |
| `issue_comment` | comments in an issue or pull request conversation |
| `pr_review` | submits a review of a pull request |
| `pr_review_comment` | comments on specific lines of a pull request's code |
| `push` | pushes commits to a branch or tag |

**Checking and scoring.** A file with a format error is not scored.

```bash
# Check the format of your output file
python scripts/evaluate.py --pred output_{studentID}.csv --input submission/test_input.csv

# Score your predictions for a validation week against your own answer file (same format)
python scripts/evaluate.py --pred YOUR_PREDICTION_FILE --truth YOUR_ANSWER_FILE
```

**Scoring.** F1 for each activity; `Score = (macro-F1 + micro-F1) / 2`.

**Reference scores.** These are the scores of five baseline methods that we tested on the
week of 2026/09/27 – 10/04.

| Method | macro-F1 | micro-F1 | Score |
|---|---|---|---|
| Random | 19.77% | 22.74% | 21.26% |
| Predict all 0 | 0.00% | 0.00% | 0.00% |
| Copy last week | 43.01% | 56.20% | 49.61% |
| GNN without time encoding | 22.80% | 48.61% | 35.70% |
| GNN with time encoding | 50.90% | 58.74% | 54.82% |

- **Copy last week**: for each activity, 1 if the user did it in the 7 days before the
  prediction time, otherwise 0.
- **GNN**: a graph neural network on the 8 tables, as in the
  [RelBench](https://relbench.stanford.edu) examples. The version with time encoding also
  tells the model how long before the prediction time each row happened.

## Data Claim

The data used in this assignment is sourced from [GH Archive](https://www.gharchive.org), a
publicly available archive of GitHub events. This assignment is intended for educational
purposes within this course.


---

## Student Implementation: Temporal Relation-Aware GNN

> This section documents the student's implementation. The course instructions above are retained for reference. The student's local validation results are **not** the official reference scores listed above.

### 1. Task, data source, and prediction period

The project predicts six GitHub activities (`open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, and `push`) for the 4,465 users listed in `submission/test_input.csv`.

- **Source:** [GH Archive](https://www.gharchive.org/). All data are processed using the course-provided `scripts/preprocess.py`, which constructs eight anonymized relational tables.
- **Historical archive:** 2026-08-16 00:00 to 2026-11-16 00:00 **Asia/Taipei** (end exclusive), corresponding to 2,208 hourly `.json.gz` files, 92 days and 199,081 cohort events identified by preprocessing.
- **Target week:** 2026-11-22 00:00 to 2026-11-29 00:00 **Asia/Taipei** (end exclusive).
- **Known limitation:** the current archive ends six days before the prediction cutoff. The pipeline does **not** invent events for the unobserved interval.

### 2. Relational graph construction

The official tables are `Users`, `Repositories`, `PullRequests`, `Issues`, `IssueComments`, `PRReviews`, `PRReviewComments`, and `Pushes` (see `SCHEMA.md`). `build_gnn_graph.py` creates `torch_geometric.data.HeteroData` snapshots with:

- Eight node types corresponding to these tables.
- Sixteen selected foreign-key relation types and sixteen reverse relation types (**32 directed edge types**).
- Snapshot-time filtering using each table's official row-time column, with strict `row_time < cutoff`.
- Simple node features (a constant and normalized age since the observed row time), plus a saved UserID-to-node-index mapping for label alignment.
- Selected relationships excluded conservatively where the linked value may reflect a later state (for example `merged_by_id` or `assignee_id`). This reduces but does **not** prove the absence of temporal leakage; remaining FK availability and approximate label definitions need review.

For the historical **2026-11-08** validation snapshot, the graph has **241,725 nodes** and **871,312 directed edges**. The **2026-11-16** snapshot has **247,143 nodes** and **890,162 directed edges**.

### 3. Multiweek training and temporal information

`build_multiweek.py` produces seven weekly training cutoffs (`2026-09-20` through `2026-11-01`), one validation cutoff (`2026-11-08`), and a provisional feature snapshot (`2026-11-16`). All dates are interpreted in Taiwan time. Its 26 features comprise:

- Per-activity historical counts for **7 and 30 days** (12 features).
- Per-activity **30-day recency** and **number of active days** (12 explicit temporal features).
- Two 30-day repository participation / peer-count features.

The training set contains **31,255 user-week rows** (4,465 users across seven cutoffs), the validation set has **4,465 user rows**, and each has six binary activity labels. Labels are locally constructed approximations using observed event timestamps; due to missing upstream events and first-seen timing, these labels should not be treated as perfect ground truth. This work uses chronological splits rather than random splitting.

### 4. Main model and ablation

**Main model:** a two-layer relation-aware GNN implemented in `train_hetero_gnn_v2.py`. It uses learnable node-type embeddings and relation-conditioned message passing, and predicts six independent binary outcomes from User-node representations.

- Loss: `BCEWithLogitsLoss` with training-only positive-class weights (capped at 15).
- Optimizer: Adam; maximum 35 epochs; early stopping based on the historical validation score.
- Decision threshold: 0.5 for each activity.
- Missing User nodes at validation/prediction time: training-label prevalence fallback.
- **Without explicit time encoding:** the same relation-aware GNN, but explicit node-age, recency, and active-days feature inputs are masked. Historical 7-/30-day counts remain, so this is **not** a completely time-agnostic model.
- **With time encoding:** includes node age, activity recency, and active-days features.

**Baseline:** `train_rf_multiweek_baseline.py` trains six Random Forest classifiers from the same multiweek tabular feature set and chronological splits. Its prediction inputs do not include graph edges. Thus, differences cannot be attributed solely to graph structure because model families and optimization methods also differ.

### 5. Local validation results

All values below are **exploratory historical validation results** for the target week **2026-11-08 to 2026-11-15 (Taiwan time)**. The same validation week has been used repeatedly during development, so the results are **not independent test performance**.

| Model | Macro-F1 | Micro-F1 | Reference score `(Macro + Micro)/2` |
|---|---:|---:|---:|
| Random Forest, 7-week baseline | 0.345964 | 0.356119 | 0.351042 |
| GNN V2, without explicit time | 0.335993 | 0.342958 | 0.339475 |
| **GNN V2, with time** | **0.347465** | 0.354785 | **0.351125** |

The with-time GNN scores slightly higher than its without-time ablation. Its difference relative to Random Forest is negligible (**0.000083** in the reference score), and this experiment does **not** establish a statistically meaningful GNN advantage. All models are evaluated on 4,465 validation users; the GNN uses a training-prevalence fallback for users absent from a graph snapshot.

### 6. Reproduction on Windows (PowerShell)

**Prerequisites:** Python 3.11 (the development environment used Python 3.11.9), enough disk space for the hourly archive, and optional NVIDIA CUDA GPU. The recorded experiment used PyTorch 2.11.0+cu128, PyTorch Geometric 2.8.1, and an RTX 5060 Ti. `requirements.txt` records the installed packages but CUDA wheels may require the corresponding PyTorch index URL.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a compatible NVIDIA GPU, install/verify a CUDA-enabled PyTorch build via the [official PyTorch selector](https://pytorch.org/get-started/locally/); for the development setup, CUDA 12.8 wheels were installed using:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install torch_geometric
```

Before downloading, verify that `scripts/download_data.py` specifies the UTC boundaries **2026-08-15 16:00 (inclusive)** to **2026-11-15 16:00 (exclusive)**. Then execute in this order from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\preprocess.py --raw raw_data --out data
.\.venv\Scripts\python.exe build_multiweek.py
.\.venv\Scripts\python.exe build_gnn_graph.py

# Model training and historical validation:
.\.venv\Scripts\python.exe train_rf_multiweek_baseline.py
.\.venv\Scripts\python.exe train_hetero_gnn_v2.py

# Target-week inference using the saved with-time GNN checkpoint:
.\.venv\Scripts\python.exe predict_gnn.py

# Official output format check:
.\.venv\Scripts\python.exe scripts\evaluate.py --pred results\output_315706002_draft.csv --input submission\test_input.csv
```

`predict_gnn.py` builds a graph and feature representation at the **2026-11-22 prediction cutoff** using **only the available historical archive ending 2026-11-16**; it loads `results/gnn_v2_with_time.pt` and produces the draft CSV. The validated submission-named copy is `results/output_315706002.csv`. If regenerating it, copy the checked draft to that filename and run the same format check on the copy.

`data/`, `raw_data/`, `.venv/` and `results/` are intentionally excluded from Git. These files are generated locally, not supplied with this repository. To reproduce inference, the training checkpoint must first be regenerated with the training command above.

### 7. Limitations and integrity notes

- **Upstream missingness:** GH Archive has missing or incomplete activity records, particularly for certain PushEvent periods; absent events may be mislabeled as negatives.
- **Approximate labels:** this implementation does not claim to reconstruct all official ground-truth events exactly.
- **Potential temporal leakage:** snapshot row-time checks are implemented, but the historical availability of every retained foreign-key value has not been proven. Historical temporal integrity must be audited before making strong causal or generalization claims.
- **Cold-start users:** 71 users are missing from the 2026-11-08 validation graph; 70 from the target-cutoff graph. They receive the fixed training-prevalence fallback.
- **Validation reuse:** model choices and stopping use the same historical validation week; an untouched holdout would provide stronger generalization evidence.
- **Prediction staleness:** the source archive stops at 2026-11-16 even though the official target starts on 2026-11-22. No missing intervening events are backfilled.

This implementation uses GH Archive for educational purposes. The instructor's reference scores above were obtained on a **different week** and should not be read as directly comparable with the student's validation scores.
