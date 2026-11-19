"""Relation-aware GNN for BigData2026 Assignment 2.

Train on seven timestamped graph snapshots, validate on 2026-11-08.
The two settings differ by explicit age/recency/active-days encodings.
Predictions here are validation-only; NOT official 2026-11-22 outputs.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path('data')
GRAPH_DIR = ROOT / 'gnn'
MULTIWEEK_DIR = ROOT / 'multiweek'
OUTPUT = Path('results')
OUTPUT.mkdir(exist_ok=True)

TARGETS = [
    'open_pr', 'open_issue', 'issue_comment',
    'pr_review', 'pr_review_comment', 'push'
]
TRAIN_DATES = [
    '2026-09-20', '2026-09-27', '2026-10-04',
    '2026-10-11', '2026-10-18', '2026-10-25', '2026-11-01'
]
VAL_DATE = '2026-11-08'
NODE_TYPES = [
    'Users', 'Repositories', 'PullRequests', 'Issues',
    'IssueComments', 'PRReviews', 'PRReviewComments', 'Pushes'
]
SEED = 42
HIDDEN = 32
EPOCHS = 35
PATIENCE = 8
MIN_EPOCHS = 10
LR = 0.003
THRESHOLD = 0.5
MAX_POS_WEIGHT = 15.0  # Controls noisy extreme weights for rare tasks.
# The historical validation week is used for checkpoint selection only;
# its score is exploratory and not an independent test result.

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE, flush=True)

train_csv = pd.read_csv(MULTIWEEK_DIR / 'train.csv', dtype={'UserID': 'string', 'cutoff': 'string'})
val_csv = pd.read_csv(MULTIWEEK_DIR / 'validation.csv', dtype={'UserID': 'string', 'cutoff': 'string'})
feature_columns = [c for c in train_csv.columns if c not in ['UserID', 'cutoff', *TARGETS]]
assert len(feature_columns) == 26, (len(feature_columns), feature_columns)
assert list(val_csv.columns) == list(train_csv.columns)
for frame in (train_csv, val_csv):
    if frame[['UserID', 'cutoff']].duplicated().any():
        raise RuntimeError('Duplicated UserID/cutoff rows')

# Pre-existing weekly count features are shared by both models; explicit
# temporal features are ablated separately. This is NOT a fully timeless model.
explicit_time_cols = [
    c for c in feature_columns if ('recency' in c or 'active_days' in c)
]
print('Feature columns:', len(feature_columns))
print('Explicit time columns:', explicit_time_cols)


def read_snapshot(cutoff: str, rows: pd.DataFrame, use_time: bool, rel_names: list):
    key = cutoff.replace('-', '')
    graph_path = GRAPH_DIR / f'graph_{key}.pt'
    mapping_path = GRAPH_DIR / f'user_mapping_{key}.csv'
    # Only load graph files you generated yourself, never unknown .pt files.
    g = torch.load(graph_path, map_location='cpu', weights_only=False)
    mapping = pd.read_csv(mapping_path, dtype={'github_user_id': 'string'})
    if mapping['graph_node_index'].duplicated().any():
        raise RuntimeError(f'Duplicate graph indices: {cutoff}')

    offsets = {}
    all_x = []
    all_type = []
    total = 0
    n_extra = len(feature_columns)
    for type_idx, ntype in enumerate(NODE_TYPES):
        x = g[ntype].x.to(torch.float32).clone()
        if not use_time:
            x[:, 1] = 0.0
        extra = torch.zeros((x.size(0), n_extra), dtype=torch.float32)
        if ntype == 'Users':
            # Unique UserID -> snapshot graph index.
            data_user = rows[['UserID', *feature_columns]].copy()
            data_user['UserID'] = data_user['UserID'].astype('string')
            joined = mapping.merge(
                data_user, left_on='github_user_id', right_on='UserID',
                how='inner', validate='one_to_one'
            )
            vals = joined[feature_columns].to_numpy(dtype=np.float32, copy=True)
            vals = np.log1p(np.maximum(vals, 0.0))
            if not use_time:
                for name in explicit_time_cols:
                    vals[:, feature_columns.index(name)] = 0.0
            if len(joined):
                extra[torch.tensor(joined['graph_node_index'].to_numpy(dtype=np.int64))] = torch.from_numpy(vals)
        offsets[ntype] = total
        total += x.size(0)
        all_x.append(torch.cat([x, extra], dim=1))
        all_type.append(torch.full((x.size(0),), type_idx, dtype=torch.long))

    x = torch.cat(all_x, dim=0)
    node_type = torch.cat(all_type)
    src_parts, dst_parts, rel_parts = [], [], []
    for etype in g.edge_types:
        if etype not in rel_names:
            raise RuntimeError(f'Unexpected relation: {etype}')
        edge = g[etype].edge_index
        src_parts.append(edge[0] + offsets[etype[0]])
        dst_parts.append(edge[1] + offsets[etype[2]])
        rel_parts.append(torch.full((edge.size(1),), rel_names.index(etype), dtype=torch.long))
    src = torch.cat(src_parts)
    dst = torch.cat(dst_parts)
    rel = torch.cat(rel_parts)

    user_lookup = pd.Series(
        mapping['graph_node_index'].to_numpy(dtype=np.int64),
        index=mapping['github_user_id'].astype('string')
    )
    row_positions = rows['UserID'].astype('string').map(user_lookup)
    observable_mask = row_positions.notna().to_numpy()
    selected_positions = np.flatnonzero(observable_mask)
    selected_node_indices = row_positions[observable_mask].to_numpy(dtype=np.int64) + offsets['Users']
    # The model only trains on user nodes visible at the historical cutoff.
    y = rows.iloc[selected_positions][TARGETS].to_numpy(dtype=np.float32, copy=True)
    ids = rows.iloc[selected_positions]['UserID'].astype('string').to_numpy(copy=True)

    print(f'{cutoff}: nodes={total:,}, directed_edges={src.numel():,}, '
          f'labeled_observed_users={len(ids):,}, omitted={len(rows)-len(ids):,}', flush=True)
    del g
    return {
        'x': x, 'node_type': node_type,
        'src': src, 'dst': dst, 'rel': rel,
        'selected_nodes': torch.from_numpy(selected_node_indices),
        'y': torch.from_numpy(y), 'ids': ids,
    }


def to_device(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


class TypedMessagePassing(nn.Module):
    def __init__(self, hidden: int, num_relations: int):
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, hidden)
        self.update = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.LayerNorm(hidden)
        )

    def forward(self, h, src, dst, rel):
        # A real relational message passing step: neighbors send learnable
        # relation-conditioned messages, then aggregate at target nodes.
        msgs = h[src] + self.relation_embedding(rel)
        aggregate = torch.zeros_like(h)
        aggregate.index_add_(0, dst, msgs)
        degree = torch.bincount(dst, minlength=h.size(0)).to(h.dtype).unsqueeze(1)
        aggregate = aggregate / degree.clamp_min(1.0)
        return self.update(torch.cat([h, aggregate], dim=-1))


class RelationalGNN(nn.Module):
    def __init__(self, input_features: int, num_relations: int):
        super().__init__()
        self.input = nn.Linear(input_features, HIDDEN)
        self.type_embedding = nn.Embedding(len(NODE_TYPES), HIDDEN)
        self.conv1 = TypedMessagePassing(HIDDEN, num_relations)
        self.conv2 = TypedMessagePassing(HIDDEN, num_relations)
        self.head = nn.Linear(HIDDEN, len(TARGETS))

    def forward(self, batch):
        h = torch.relu(self.input(batch['x']) + self.type_embedding(batch['node_type']))
        h = self.conv1(h, batch['src'], batch['dst'], batch['rel'])
        h = self.conv2(h, batch['src'], batch['dst'], batch['rel'])
        return self.head(h[batch['selected_nodes']])


# Confirm all snapshots use identical edge-type definitions/order.
first_graph = torch.load(GRAPH_DIR / 'graph_20260920.pt', map_location='cpu', weights_only=False)
REL_NAMES = list(first_graph.edge_types)
assert len(REL_NAMES) == 32, len(REL_NAMES)
del first_graph


def metrics(y_true: np.ndarray, y_hat: np.ndarray):
    macro = f1_score(y_true, y_hat, average='macro', zero_division=0)
    micro = f1_score(y_true, y_hat, average='micro', zero_division=0)
    return macro, micro, (macro + micro) / 2


def validation_predictions(model, batch_cpu, cutoff_frame):
    """Score the complete 4,465-user validation cohort in original order.

    A user missing from the snapshot graph receives the training-set
    prevalence probability. This is a simple cold-start fallback, and
    deliberately does not use validation labels.
    """
    model.eval()
    with torch.no_grad():
        batch = to_device(batch_cpu)
        observed_probs = torch.sigmoid(model(batch)).cpu().numpy()
        del batch

    result = pd.DataFrame({'UserID': cutoff_frame['UserID'].astype('string')})
    observed = pd.DataFrame(observed_probs, columns=TARGETS)
    observed.insert(0, 'UserID', pd.Series(batch_cpu['ids'], dtype='string'))
    result = result.merge(observed, on='UserID', how='left', validate='one_to_one')
    missing = result[TARGETS].isna().all(axis=1)
    for target in TARGETS:
        result[target] = result[target].fillna(float(TRAIN_PREVALENCE[target]))
    if result[TARGETS].isna().any().any():
        raise RuntimeError('Unresolved user predictions')
    return result, int(missing.sum())


# Weights estimated strictly from the training periods (no validation leakage).
TRAIN_PREVALENCE = train_csv[TARGETS].mean().clip(0.0001, 0.9999)
num_positive = train_csv[TARGETS].sum().astype(float)
num_negative = len(train_csv) - num_positive
weight_values = (num_negative / num_positive.clip(lower=1)).clip(1.0, MAX_POS_WEIGHT)
POS_WEIGHT = torch.tensor(weight_values.to_numpy(dtype=np.float32, copy=True), device=DEVICE)
print('Training prevalence:', TRAIN_PREVALENCE.round(4).to_dict(), flush=True)
print('Positive-class weights:', dict(zip(TARGETS, POS_WEIGHT.cpu().tolist())), flush=True)


def run_experiment(use_time: bool):
    name = 'with_time' if use_time else 'without_explicit_time'
    print(f'\n==== Training: {name} ====', flush=True)
    torch.manual_seed(SEED)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    model = RelationalGNN(2 + len(feature_columns), len(REL_NAMES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    best_score, best_state, best_epoch = -1.0, None, 0
    epochs_without_improvement = 0
    history = []

    # Cache graph inputs only in CPU RAM; move one snapshot at a time to GPU.
    train_batches = {}
    for cutoff in TRAIN_DATES:
        subset = train_csv[train_csv['cutoff'] == cutoff].copy()
        assert len(subset) == 4465, (cutoff, len(subset))
        train_batches[cutoff] = read_snapshot(cutoff, subset, use_time, REL_NAMES)
    val_batch = read_snapshot(VAL_DATE, val_csv, use_time, REL_NAMES)
    actual_all = val_csv[TARGETS].to_numpy(dtype=np.int32, copy=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for cutoff in TRAIN_DATES:
            batch = to_device(train_batches[cutoff])
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, batch['y'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            train_losses.append(loss.item())
            del batch, logits, loss

        prob_df, missing = validation_predictions(model, val_batch, val_csv)
        probs = prob_df[TARGETS].to_numpy(dtype=np.float32, copy=True)
        pred = (probs >= THRESHOLD).astype(np.int32)
        macro, micro, score = metrics(actual_all, pred)
        item = {'Model': name, 'Epoch': epoch, 'Loss': float(np.mean(train_losses)),
                'Macro_F1': macro, 'Micro_F1': micro, 'Score': score,
                'Missing_Users': missing}
        history.append(item)
        print(f"Epoch {epoch:02d} loss={item['Loss']:.4f} "
              f'macro={macro:.4f} micro={micro:.4f} score={score:.4f} '
              f'cold_start={missing}', flush=True)

        if score > best_score + 1e-5:
            best_score, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= MIN_EPOCHS and epochs_without_improvement >= PATIENCE:
            print(f'Early stopping at epoch {epoch}; best epoch {best_epoch}', flush=True)
            break

    model.load_state_dict(best_state)
    prob_df, missing = validation_predictions(model, val_batch, val_csv)
    probs = prob_df[TARGETS].to_numpy(dtype=np.float32, copy=True)
    pred = (probs >= THRESHOLD).astype(np.int32)
    macro, micro, score = metrics(actual_all, pred)
    print(f'BEST epoch={best_epoch}, macro={macro:.4f}, micro={micro:.4f}, '
          f'score={score:.4f}, cold_start={missing}', flush=True)

    rows = []
    for i, target in enumerate(TARGETS):
        p = precision_score(actual_all[:, i], pred[:, i], zero_division=0)
        r = recall_score(actual_all[:, i], pred[:, i], zero_division=0)
        f = f1_score(actual_all[:, i], pred[:, i], zero_division=0)
        rows.append({'Model': name, 'Activity': target, 'Precision': p,
                     'Recall': r, 'F1': f, 'Actual_Positive': int(actual_all[:, i].sum()),
                     'Predicted_Positive': int(pred[:, i].sum())})
        print(f'{target:20s} P={p:.4f} R={r:.4f} F1={f:.4f}', flush=True)

    torch.save({'model_state': best_state, 'feature_columns': feature_columns,
                'relation_names': REL_NAMES, 'node_types': NODE_TYPES,
                'use_time': use_time, 'hidden': HIDDEN, 'epoch': best_epoch,
                'threshold': THRESHOLD, 'pos_weight': POS_WEIGHT.cpu(),
                'train_prevalence': TRAIN_PREVALENCE.to_dict()},
               OUTPUT / f'gnn_v2_{name}.pt')

    predictions = prob_df[['UserID']].copy()
    for i, target in enumerate(TARGETS):
        predictions[f'{target}_prob'] = probs[:, i]
        predictions[target] = pred[:, i]
    assert len(predictions) == len(val_csv) == 4465
    predictions.to_csv(OUTPUT / f'gnn_v2_{name}_validation_predictions.csv', index=False)
    return rows, history, {'Model': name, 'Best_Epoch': best_epoch, 'Macro_F1': macro,
                           'Micro_F1': micro, 'Reference_Score': score,
                           'Missing_Users': missing}


if __name__ == '__main__':
    all_rows, all_history, summary = [], [], []
    for use_time in (False, True):
        rows, history, report = run_experiment(use_time)
        all_rows.extend(rows)
        all_history.extend(history)
        summary.append(report)
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    pd.DataFrame(all_rows).to_csv(OUTPUT / 'gnn_v2_ablation_details.csv', index=False)
    pd.DataFrame(all_history).to_csv(OUTPUT / 'gnn_v2_training_history.csv', index=False)
    pd.DataFrame(summary).to_csv(OUTPUT / 'gnn_v2_summary.csv', index=False)
    print('\nFINAL SUMMARY', flush=True)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print('\nGNN v2 experiments completed. Results under results/.', flush=True)
    print('These are exploratory validation results, NOT official test-week predictions.', flush=True)
