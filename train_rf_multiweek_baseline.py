"""Seven-week Random Forest baseline for BigData2026 Assignment 2.

Inputs:
    data/multiweek/train.csv
    data/multiweek/validation.csv
Outputs:
    results/rf_multiweek_summary.csv
    results/rf_multiweek_details.csv
    results/rf_multiweek_validation_predictions.csv
    results/rf_multiweek_feature_importance.csv

This is an exploratory time-based validation, not official test performance.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path('data/multiweek')
OUTPUT = Path('results')
OUTPUT.mkdir(parents=True, exist_ok=True)
TARGETS = [
    'open_pr', 'open_issue', 'issue_comment',
    'pr_review', 'pr_review_comment', 'push'
]
EXPECTED_TRAIN_DATES = [
    '2026-09-20', '2026-09-27', '2026-10-04',
    '2026-10-11', '2026-10-18', '2026-10-25', '2026-11-01'
]
VALIDATION_CUTOFF = '2026-11-08'


def load(path):
    frame = pd.read_csv(path, dtype={'UserID': 'string', 'cutoff': 'string'})
    required = {'UserID', 'cutoff', *TARGETS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'{path}: missing columns: {sorted(missing)}')
    if frame[['UserID', 'cutoff']].duplicated().any():
        raise ValueError(f'{path}: duplicate UserID/cutoff pairs')
    return frame


train = load(ROOT / 'train.csv')
validation = load(ROOT / 'validation.csv')
feature_cols = [c for c in train.columns if c not in ['UserID', 'cutoff', *TARGETS]]
if len(feature_cols) != 26:
    raise ValueError(f'Expected 26 features, got {len(feature_cols)}')
if list(validation.columns) != list(train.columns):
    raise ValueError('Train/validation column order is inconsistent')
if sorted(train['cutoff'].unique().tolist()) != EXPECTED_TRAIN_DATES:
    raise ValueError('Unexpected training cutoffs')
if validation['cutoff'].nunique() != 1 or validation['cutoff'].iloc[0] != VALIDATION_CUTOFF:
    raise ValueError('Unexpected validation cutoff')
if len(train) != 31255 or len(validation) != 4465:
    raise ValueError('Unexpected dataset row counts')

# The same users can appear across several training weeks: each row is a user-week.
# Features were computed using history before the per-row cutoff.
X_train = train[feature_cols].to_numpy(dtype=np.float32, copy=True)
X_val = validation[feature_cols].to_numpy(dtype=np.float32, copy=True)
y_train = train[TARGETS].to_numpy(dtype=np.int32, copy=True)
y_val = validation[TARGETS].to_numpy(dtype=np.int32, copy=True)
if not (np.isfinite(X_train).all() and np.isfinite(X_val).all()):
    raise ValueError('Non-finite input features detected')
if not np.isin(y_train, [0, 1]).all() or not np.isin(y_val, [0, 1]).all():
    raise ValueError('Labels must be binary')

print('Train shape:', X_train.shape)
print('Validation shape:', X_val.shape)
print('Training cutoffs:', ', '.join(EXPECTED_TRAIN_DATES))
print('Validation cutoff:', VALIDATION_CUTOFF)
print('Feature count:', len(feature_cols))
print('Classifier: RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, max_features="sqrt", class_weight="balanced_subsample", random_state=42)')
print('Decision threshold: 0.5')

predictions = np.zeros_like(y_val)
probabilities = np.zeros(y_val.shape, dtype=np.float64)
importance = []
rows = []

for i, target in enumerate(TARGETS):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
    )
    if np.unique(y_train[:, i]).size < 2:
        raise ValueError(f'Only one class in training for {target}')
    model.fit(X_train, y_train[:, i])
    probs = model.predict_proba(X_val)[:, list(model.classes_).index(1)]
    preds = (probs >= 0.5).astype(np.int32)
    predictions[:, i] = preds
    probabilities[:, i] = probs
    p = precision_score(y_val[:, i], preds, zero_division=0)
    r = recall_score(y_val[:, i], preds, zero_division=0)
    f1 = f1_score(y_val[:, i], preds, zero_division=0)
    rows.append({
        'Model': 'RandomForest_7week', 'Activity': target,
        'Precision': p, 'Recall': r, 'F1': f1,
        'Actual_Positive': int(y_val[:, i].sum()),
        'Predicted_Positive': int(preds.sum()),
    })
    for col, value in zip(feature_cols, model.feature_importances_):
        importance.append({'Activity': target, 'Feature': col, 'Importance': float(value)})
    print(f'{target:20s} P={p:.4f} R={r:.4f} F1={f1:.4f} Actual+={y_val[:, i].sum():4d} Predicted+={preds.sum():4d}', flush=True)

macro = f1_score(y_val, predictions, average='macro', zero_division=0)
micro = f1_score(y_val, predictions, average='micro', zero_division=0)
score = (macro + micro) / 2
summary = pd.DataFrame([{
    'Model': 'RandomForest_7week', 'Train_UserWeeks': len(train),
    'Validation_Users': len(validation), 'Features': len(feature_cols),
    'Macro_F1': macro, 'Micro_F1': micro, 'Reference_Score': score,
}])
summary.to_csv(OUTPUT / 'rf_multiweek_summary.csv', index=False)
pd.DataFrame(rows).to_csv(OUTPUT / 'rf_multiweek_details.csv', index=False)
pd.DataFrame(importance).to_csv(OUTPUT / 'rf_multiweek_feature_importance.csv', index=False)
pred_file = validation[['UserID']].copy()
for i, target in enumerate(TARGETS):
    pred_file[f'{target}_prob'] = probabilities[:, i]
    pred_file[target] = predictions[:, i]
pred_file.to_csv(OUTPUT / 'rf_multiweek_validation_predictions.csv', index=False)
print('\n=== RANDOM FOREST 7-WEEK SUMMARY ===')
print(summary.to_string(index=False))
print('\nFor comparison only: GNN V2 with_time historical validation reference score = 0.351125.')
print('Both use the same weekly train/validation cutoffs and 4,465 validation users.')
print('GNN additionally uses graph relations and explicit time features; Random Forest uses 26 tabular features.')
print('Scores are exploratory on a repeatedly used validation week, not official test results.')
print('Completed. Results saved under results/.')
