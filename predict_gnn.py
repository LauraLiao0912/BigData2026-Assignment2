"""Inference for BigData2026 Assignment 2, using the saved GNN V2 checkpoint.

Constructs a genuine 2026-11-22 00:00 Asia/Taipei feature and graph cutoff,
using only records already stored in data/db. No future data is fabricated.
Run from the repository root with its Python venv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

from build_gnn_graph import build_graph

ROOT = Path('data')
DB = ROOT / 'db'
GRAPH_DIR = ROOT / 'gnn'
OUTPUT = Path('results')
OUTPUT.mkdir(parents=True, exist_ok=True)
CUTOFF_DATE = '2026-11-22'
CUTOFF = pd.Timestamp(CUTOFF_DATE, tz='Asia/Taipei').tz_convert('UTC')
TARGETS = ['open_pr', 'open_issue', 'issue_comment', 'pr_review', 'pr_review_comment', 'push']
NODE_TYPES = ['Users', 'Repositories', 'PullRequests', 'Issues', 'IssueComments', 'PRReviews', 'PRReviewComments', 'Pushes']
CONFIG = {
    'open_pr': ('PullRequests', 'author_id', 'created_at', 'first_seen_at'),
    'open_issue': ('Issues', 'author_id', 'created_at', 'first_seen_at'),
    'issue_comment': ('IssueComments', 'author_id', 'created_at', 'created_at'),
    'pr_review': ('PRReviews', 'reviewer_id', 'submitted_at', 'first_seen_at'),
    'pr_review_comment': ('PRReviewComments', 'author_id', 'created_at', 'created_at'),
    'push': ('Pushes', 'user_id', 'created_at', 'created_at'),
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TypedMessagePassing(nn.Module):
    def __init__(self, hidden, num_relations):
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, hidden)
        self.update = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.LayerNorm(hidden))

    def forward(self, h, src, dst, rel):
        msgs = h[src] + self.relation_embedding(rel)
        aggregate = torch.zeros_like(h)
        aggregate.index_add_(0, dst, msgs)
        degree = torch.bincount(dst, minlength=h.size(0)).to(h.dtype).unsqueeze(1)
        aggregate = aggregate / degree.clamp_min(1.0)
        return self.update(torch.cat([h, aggregate], dim=-1))


class RelationalGNN(nn.Module):
    def __init__(self, input_features, num_relations, hidden):
        super().__init__()
        self.input = nn.Linear(input_features, hidden)
        self.type_embedding = nn.Embedding(len(NODE_TYPES), hidden)
        self.conv1 = TypedMessagePassing(hidden, num_relations)
        self.conv2 = TypedMessagePassing(hidden, num_relations)
        self.head = nn.Linear(hidden, len(TARGETS))

    def forward(self, batch):
        h = torch.relu(self.input(batch['x']) + self.type_embedding(batch['node_type']))
        h = self.conv1(h, batch['src'], batch['dst'], batch['rel'])
        h = self.conv2(h, batch['src'], batch['dst'], batch['rel'])
        return self.head(h[batch['selected_nodes']])


def prepare_final_features(test):
    """Replicate build_multiweek.py feature definitions at the true target cutoff."""
    users = pd.read_parquet(DB / 'Users.parquet')
    user_map = (users[users['is_cohort'].fillna(False)]
                .set_index('user_id')['github_user_id'].astype('string'))
    result = test[['UserID']].copy()
    result['cutoff'] = CUTOFF_DATE
    relational_parts = []
    start_30d = CUTOFF - pd.Timedelta(days=30)
    start_7d = CUTOFF - pd.Timedelta(days=7)
    for target, (table, user_col, _event_col, seen_col) in CONFIG.items():
        df = pd.read_parquet(DB / f'{table}.parquet')
        if target == 'open_issue':
            df = df[df['is_pull_request'].eq(False).fillna(False)].copy()
        df['UserID'] = df[user_col].map(user_map)
        df['seen_time'] = pd.to_datetime(df[seen_col], utc=True, errors='coerce')
        history = df[(df['seen_time'] >= start_30d) &
                     (df['seen_time'] < CUTOFF)].copy()
        for days, start in [(7, start_7d), (30, start_30d)]:
            counts = history[history['seen_time'] >= start].groupby('UserID').size()
            result[f'{target}_count_{days}d'] = result['UserID'].map(counts).fillna(0).astype(int)
        last_seen = history.groupby('UserID')['seen_time'].max()
        recency = (CUTOFF - last_seen).dt.total_seconds().div(86400)
        result[f'{target}_recency_30d'] = result['UserID'].map(recency).fillna(31).clip(0, 31)
        history['local_date'] = history['seen_time'].dt.tz_convert('Asia/Taipei').dt.date
        active_days = history.groupby('UserID')['local_date'].nunique()
        result[f'{target}_active_days_30d'] = result['UserID'].map(active_days).fillna(0).astype(int)
        relational_parts.append(history[['UserID', 'repo_id']])
    pairs = pd.concat(relational_parts, ignore_index=True).dropna(subset=['repo_id']).drop_duplicates()
    repo_users = pairs.groupby('repo_id')['UserID'].nunique()
    pairs['peer_count'] = pairs['repo_id'].map(repo_users) - 1
    unique_repos = pairs.groupby('UserID')['repo_id'].nunique()
    max_peers = pairs.groupby('UserID')['peer_count'].max()
    result['unique_repos_30d'] = result['UserID'].map(unique_repos).fillna(0).astype(int)
    result['max_repo_peers_30d'] = result['UserID'].map(max_peers).fillna(0).astype(int)
    return result


def prepare_graph_batch(g, mapping, rows, feature_columns, relation_names, use_time):
    offsets, all_x, all_type = {}, [], []
    total = 0
    for type_idx, ntype in enumerate(NODE_TYPES):
        x = g[ntype].x.to(torch.float32).clone()
        if not use_time:
            x[:, 1] = 0.0
        extra = torch.zeros((x.shape[0], len(feature_columns)), dtype=torch.float32)
        if ntype == 'Users':
            joined = mapping.merge(rows[['UserID', *feature_columns]],
                                   left_on='github_user_id', right_on='UserID',
                                   how='inner', validate='one_to_one')
            vals = joined[feature_columns].to_numpy(dtype=np.float32, copy=True)
            vals = np.log1p(np.maximum(vals, 0.0))
            if not use_time:
                for name in feature_columns:
                    if 'recency' in name or 'active_days' in name:
                        vals[:, feature_columns.index(name)] = 0.0
            if len(joined):
                extra[torch.from_numpy(joined['graph_node_index'].to_numpy(dtype=np.int64, copy=True))] = torch.from_numpy(vals)
        offsets[ntype] = total
        total += x.shape[0]
        all_x.append(torch.cat([x, extra], dim=1))
        all_type.append(torch.full((x.shape[0],), type_idx, dtype=torch.long))
    if set(g.edge_types) != set(relation_names):
        raise RuntimeError('Graph relation types do not match saved model')
    rel_index = {name: i for i, name in enumerate(relation_names)}
    src_parts, dst_parts, rel_parts = [], [], []
    for etype in g.edge_types:
        edge = g[etype].edge_index
        src_parts.append(edge[0] + offsets[etype[0]])
        dst_parts.append(edge[1] + offsets[etype[2]])
        rel_parts.append(torch.full((edge.shape[1],), rel_index[etype], dtype=torch.long))
    lookup = pd.Series(mapping['graph_node_index'].to_numpy(dtype=np.int64),
                       index=mapping['github_user_id'].astype('string'))
    positions = rows['UserID'].map(lookup)
    present = positions.notna().to_numpy()
    selected_nodes = positions[present].to_numpy(dtype=np.int64, copy=True) + offsets['Users']
    selected_ids = rows.loc[present, 'UserID'].reset_index(drop=True)
    return {
        'x': torch.cat(all_x), 'node_type': torch.cat(all_type),
        'src': torch.cat(src_parts), 'dst': torch.cat(dst_parts),
        'rel': torch.cat(rel_parts),
        'selected_nodes': torch.from_numpy(selected_nodes)
    }, selected_ids


def main():
    print('Device:', DEVICE)
    print('Prediction target: 2026-11-22 00:00 to 2026-11-29 00:00 Asia/Taipei')
    print('Historical source: stored data/db (through 2026-11-16 00:00 Asia/Taipei)')
    test = pd.read_csv('submission/test_input.csv', dtype={'UserID': 'string'})
    assert test['UserID'].notna().all() and test['UserID'].is_unique
    rows = prepare_final_features(test)
    assert not rows.isna().any().any()
    checkpoint = torch.load(OUTPUT / 'gnn_v2_with_time.pt', map_location='cpu', weights_only=True)
    feature_columns = checkpoint['feature_columns']
    relation_names = [tuple(e) for e in checkpoint['relation_names']]
    assert checkpoint['use_time'] is True
    assert set(feature_columns) == set(rows.columns) - {'UserID', 'cutoff'}
    rows = rows[['UserID', 'cutoff', *feature_columns]]
    assert len(rows) == len(test)
    rows.to_csv(ROOT / 'multiweek' / 'final_20261122.csv', index=False)
    graph_path = GRAPH_DIR / 'graph_20261122.pt'
    mapping_path = GRAPH_DIR / 'user_mapping_20261122.csv'
    if not graph_path.exists() or not mapping_path.exists():
        print('Building graph snapshot for 2026-11-22 ...', flush=True)
        build_graph(CUTOFF_DATE)
    # Only load a graph file produced locally by the project's own build_gnn_graph.py.
    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    mapping = pd.read_csv(mapping_path, dtype={'github_user_id': 'string'})
    if mapping['github_user_id'].duplicated().any() or mapping['graph_node_index'].duplicated().any():
        raise RuntimeError('Duplicate graph user mapping')
    batch, observed_ids = prepare_graph_batch(graph, mapping, rows, feature_columns, relation_names, True)
    model = RelationalGNN(2 + len(feature_columns), len(relation_names), int(checkpoint['hidden']))
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model = model.to(DEVICE).eval()
    print(f'Graph nodes: {len(batch["node_type"]):,}; graph users predicted: {len(observed_ids):,}')
    with torch.no_grad():
        batch_gpu = {k: v.to(DEVICE) for k, v in batch.items()}
        probs = torch.sigmoid(model(batch_gpu)).cpu().numpy()
    pred_observed = pd.DataFrame(probs, columns=TARGETS)
    pred_observed.insert(0, 'UserID', observed_ids)
    result = test[['UserID']].merge(pred_observed, on='UserID', how='left', validate='one_to_one')
    cold_start = result[TARGETS].isna().all(axis=1)
    for t in TARGETS:
        result[t] = result[t].fillna(float(checkpoint['train_prevalence'][t]))
    assert not result[TARGETS].isna().any().any()
    threshold = float(checkpoint['threshold'])
    probability_report = result.rename(columns={t: f'{t}_prob' for t in TARGETS})
    probability_report.to_csv(OUTPUT / 'gnn_v2_20261122_probabilities.csv', index=False)
    for t in TARGETS:
        result[t] = (result[t] >= threshold).astype(int)
    assert len(result) == len(test)
    assert result['UserID'].tolist() == test['UserID'].tolist()
    assert list(result.columns) == ['UserID', *TARGETS]
    assert result[TARGETS].isin([0, 1]).all().all()
    output = OUTPUT / 'output_315706002_draft.csv'
    result.to_csv(output, index=False)
    print(f'Cold-start users: {int(cold_start.sum())}')
    print('Predicted positives:')
    print(result[TARGETS].sum().to_string())
    print('Saved:', output)
    print('NOTE: Uses available history only through 2026-11-16; the six-day gap is unobserved.')
    print('Run scripts/evaluate.py to validate format before submission.')


if __name__ == '__main__':
    main()
