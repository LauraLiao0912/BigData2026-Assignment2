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

## Student Implementation: GitHub User Activity Prediction

### 1. Project Overview

This project predicts six GitHub user activities over a seven-day period using historical events from GH Archive.

The six target activities are `open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, and `push`.

### 2. Data Source

The dataset is obtained from **GH Archive**: https://www.gharchive.org/.

The official `scripts/preprocess.py` is used to construct eight anonymized relational tables.

The current experimental dataset covers October 25 to November 16, 2026 (Taiwan time, end exclusive), consisting of 528 hourly GH Archive files.

Some upstream events, particularly PushEvent records, are missing. These missing events cannot be recovered from the permitted dataset.

### 3. Environment Setup

The project was developed using Python 3.14.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### 4. Data Preparation

Download historical GH Archive files:

```bash
python scripts/download_data.py
```

Generate the eight relational tables using the official preprocessing script:

```bash
python scripts/preprocess.py --raw raw_data --out data
```

### 5. Feature Engineering

Three feature groups are implemented:

- **Historical Activity Features (22):** Activity counts over 1, 3, and 7 days, total activity counts, and recent activity indicators.
- **Relational Features (4):** Repository participation, shared repository activity, and peer activity statistics.
- **Temporal Features (18):** Activity recency, number of active days, and recent activity trends.

Generate features and historical labels:

```bash
python build_labels.py
python build_features.py
python build_relational_features.py
python build_temporal_features.py
```

### 6. Modeling and Experiments

Two classification approaches are evaluated:

- Logistic Regression
- Random Forest

The experiments compare four feature configurations: Baseline, Baseline + Relational, Baseline + Temporal, and Full Features.

Training and validation are separated chronologically rather than randomly.

The reported validation results are exploratory and may be affected by missing upstream events and repeated model selection on the same validation period.

### 7. Reproduce the Experiments

```bash
python train_baseline.py
python experiment_threshold.py
python experiment_relational.py
python experiment_full_ablation.py
python experiment_random_forest.py
```

### 8. Generate Predictions

The current prediction model uses Random Forest with 22 historical features and 18 temporal features.

Run:

```bash
python train_final_model.py
```

This generates:

```text
results/output_draft.csv
```

Validate the output format:

```bash
python scripts/evaluate.py --pred results/output_draft.csv --input submission/test_input.csv
```

The current draft uses historical data available through November 16, 2026. It is a preliminary prediction file and must be reviewed before final submission.

### 9. Current Experimental Results

| Model | Features | Macro-F1 | Micro-F1 | Reference Score |
|---|---:|---:|---:|---:|
| Logistic Regression + Historical | 22 | 0.2458 | 0.2530 | 0.2494 |
| Logistic Regression + Relational | 26 | 0.2491 | 0.2550 | 0.2521 |
| Random Forest + Historical | 22 | 0.2631 | 0.2713 | 0.2672 |
| Random Forest + Relational | 26 | 0.2672 | 0.2746 | 0.2709 |
| Random Forest + Temporal | 40 | 0.2921 | 0.3005 | **0.2963** |
| Random Forest + Full | 44 | 0.2893 | 0.2982 | 0.2938 |

These scores were obtained on historical validation labels for November 8–15, 2026, and do not represent official test-week performance.

### 10. Limitations

- Missing GH Archive events may introduce label noise and temporal bias.
- Some users have no observed activity history (cold-start problem).
- First-observed object timestamps are not always equivalent to original event timestamps.
- The model has been evaluated on a limited historical validation period.
- Further temporal holdout testing is needed to assess generalization.