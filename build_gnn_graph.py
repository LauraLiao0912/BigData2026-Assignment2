import numpy as np
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

DB = Path("data/db")
OUT = Path("data/gnn")
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Table definitions
# Primary Key and Row Time
# --------------------------------------------------

TABLES = {
    "Users": ("user_id", "first_seen_at"),
    "Repositories": ("repo_id", "first_seen_at"),
    "PullRequests": ("pr_id", "first_seen_at"),
    "Issues": ("issue_id", "first_seen_at"),
    "IssueComments": ("comment_id", "created_at"),
    "PRReviews": ("review_id", "first_seen_at"),
    "PRReviewComments": ("comment_id", "created_at"),
    "Pushes": ("push_id", "created_at"),
}

# --------------------------------------------------
# 3. Foreign Key relationships
# Exclude potentially future-dependent relationships
# --------------------------------------------------

RELATIONS = [
    # PullRequests
    ("PullRequests", "repo_id", "Repositories"),
    ("PullRequests", "author_id", "Users"),

    # Issues
    ("Issues", "repo_id", "Repositories"),
    ("Issues", "author_id", "Users"),

    # IssueComments
    ("IssueComments", "issue_id", "Issues"),
    ("IssueComments", "repo_id", "Repositories"),
    ("IssueComments", "author_id", "Users"),

    # PRReviews
    ("PRReviews", "pr_id", "PullRequests"),
    ("PRReviews", "repo_id", "Repositories"),
    ("PRReviews", "reviewer_id", "Users"),

    # PRReviewComments
    ("PRReviewComments", "pr_id", "PullRequests"),
    ("PRReviewComments", "repo_id", "Repositories"),
    ("PRReviewComments", "author_id", "Users"),
    (
        "PRReviewComments",
        "pull_request_review_id",
        "PRReviews"
    ),

    # Pushes
    ("Pushes", "user_id", "Users"),
    ("Pushes", "repo_id", "Repositories"),
]

# Excluded relationships:
# PullRequests.merged_by_id
# Issues.assignee_id
# Issues.pull_request_id
# PRReviewComments.in_reply_to_id

# --------------------------------------------------
# 4. Convert Taiwan Time to UTC
# --------------------------------------------------

def to_utc(date_str):
    return pd.Timestamp(
        date_str,
        tz="Asia/Taipei"
    ).tz_convert("UTC")


# --------------------------------------------------
# 5. Build graph snapshot
# --------------------------------------------------

def build_graph(cutoff_date):

    cutoff = to_utc(cutoff_date)

    data = HeteroData()

    tables = {}
    id_maps = {}

    print(f"\nGraph cutoff: {cutoff_date}")

    # ----------------------------------------------
    # Step A: Build nodes
    # ----------------------------------------------

    for name, (pk, time_col) in TABLES.items():

        df = pd.read_parquet(
            DB / f"{name}.parquet"
        )

        row_time = pd.to_datetime(
            df[time_col],
            utc=True,
            errors="coerce"
        )

        # Only include nodes observed before cutoff
        mask = (
            row_time.notna() &
            (row_time < cutoff)
        )

        df = df.loc[mask].copy()
        row_time = row_time.loc[mask]

        df = df.reset_index(drop=True)
        row_time = row_time.reset_index(drop=True)

        # Original Primary Key -> Graph node index
        id_maps[name] = pd.Series(
            range(len(df)),
            index=df[pk].astype("int64")
        )

        if not id_maps[name].index.is_unique:
            raise ValueError(
                f"Duplicate primary keys in {name}"
            )

        # Node age in days
        age_days = (
            (cutoff - row_time)
            .dt.total_seconds()
            .div(86400)
            .clip(lower=0, upper=365)
        )

        # Basic node features:
        # [constant, normalized age]
        node_features = torch.column_stack([
            torch.ones(
                len(df),
                dtype=torch.float32
            ),
            torch.tensor(
                age_days.to_numpy(),
                dtype=torch.float32
            ) / 365.0
        ])

        data[name].x = node_features
        data[name].num_nodes = len(df)

        # Keep original row IDs for later alignment
        data[name].node_id = torch.tensor(
            df[pk].to_numpy(dtype="int64"),
            dtype=torch.long
        )

        tables[name] = df

        print(
            f"{name:20s}: "
            f"{len(df):>8,} nodes"
        )

    # ----------------------------------------------
    # Step B: Build edges
    # ----------------------------------------------

    print("\nBuilding edges:")

    for source, fk, target in RELATIONS:

        source_df = tables[source]
        target_map = id_maps[target]

        # Map FK values to graph node indices
        target_ids = source_df[fk].map(
            target_map
        )

        valid = target_ids.notna()

        src_indices = (
            source_df.index[valid]
            .to_numpy(dtype="int64")
        )

        dst_indices = (
            target_ids[valid]
            .to_numpy(dtype="int64")
        )

        edge_index = torch.from_numpy(
            np.stack([src_indices, dst_indices])
        ).long()

        relation_name = f"fk_{fk}"

        # Forward edge
        data[
            source,
            relation_name,
            target
        ].edge_index = edge_index

        # Reverse edge
        data[
            target,
            f"rev_{source}_{fk}",
            source
        ].edge_index = edge_index.flip(0).contiguous()

        print(
            f"{source}.{fk} -> {target}: "
            f"{edge_index.shape[1]:,} edges"
        )

    # ----------------------------------------------
    # Step C: Save graph and node mapping
    # ----------------------------------------------

    date_str = cutoff_date.replace("-", "")

    graph_path = OUT / f"graph_{date_str}.pt"

    torch.save(
        data,
        graph_path
    )

    # Save user ID mapping for training labels
    users_df = tables["Users"]

    user_mapping = users_df[
        ["user_id", "github_user_id"]
    ].copy()

    user_mapping.insert(
        0,
        "graph_node_index",
        range(len(user_mapping))
    )

    mapping_path = OUT / (
        f"user_mapping_{date_str}.csv"
    )

    user_mapping.to_csv(
        mapping_path,
        index=False
    )

    print(f"\nSaved graph: {graph_path}")
    print(f"Saved user mapping: {mapping_path}")

    # ----------------------------------------------
    # Step D: Summary
    # ----------------------------------------------

    total_nodes = sum(
        data[t].num_nodes
        for t in data.node_types
    )

    total_edges = sum(
        data[e].edge_index.shape[1]
        for e in data.edge_types
    )

    print("\nGraph Summary")
    print(f"Node types: {len(data.node_types)}")
    print(f"Edge types: {len(data.edge_types)}")
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total directed edges: {total_edges:,}")

    return data


# --------------------------------------------------
# 6. Build weekly snapshots
# --------------------------------------------------

if __name__ == "__main__":

    CUTOFFS = [
        # Training weeks
        "2026-09-20",
        "2026-09-27",
        "2026-10-04",
        "2026-10-11",
        "2026-10-18",
        "2026-10-25",
        "2026-11-01",

        # Validation
        "2026-11-08",

        # Draft final prediction
        "2026-11-16",
    ]

    for i, cutoff in enumerate(CUTOFFS, start=1):

        print("\n" + "=" * 60)
        print(
            f"Building graph {i}/{len(CUTOFFS)}"
        )
        print("=" * 60)

        graph = build_graph(cutoff)

        del graph

    print("\n" + "=" * 60)
    print("All graph snapshots completed.")
    print(f"Total snapshots: {len(CUTOFFS)}")
    print(f"Output directory: {OUT}")
    print("=" * 60)
