"""Shared code for the FT-Transformer teammate's runs in the team study (references/Team_plan_graph_vs_V.pdf):
paths, the two shared input files, preprocessing, the shared graph, scoring and saving. No experiments here.

Team plan / paper -> this file
  everyone loads Reset's split file, nobody makes their own split  -> load_split()   (hash-checked)
  column set A = paper Table 2 (390 columns), B = A without V      -> column_set()
  paper 3.2: medians, mean/std and categories from training only   -> load_features()
  the SOA member's entity graph files, nobody builds their own     -> load_graph()   (read straight from the zip)
  ROC-AUC and PR-AUC (average precision), the paper's two metrics  -> evaluate()
  three hand-in files + one results-table row per run              -> save_run(), write_once()
  data and graph checks before any training                        -> data_checks(), graph_checks()
"""
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
SPLIT = DATA / "ieee_sorted_split.parquet"       # Reset's split file: the ONE shared split (do not change it)
GRAPH = DATA / "graph_files.zip"                 # the SOA member's graph files, read without unzipping
SHA256 = {SPLIT.name: "9cde42fd5c67ed7edb7f76ab7f8b3d7d7e2c4e789cb960efc5c466182875dbe1",
          GRAPH.name: "5d0d5f6534da40fc1d0da3205ee158c54f2320ef2c2c3814a8b70ae63031182b"}  # as received 2026-11-09
OUT = ROOT / "results" / "graph_vs_V"            # every run's outputs (style guide 6)
RUNS_CSV = OUT / "runs.csv"
CODE = ["common.py", "models.py", "runs.py"]     # the code behind a result; hashed into every history file

SPLITS = ["train", "val", "test"]                # the split file's own labels, in time order
SPLIT_SIZES = {"train": 413378, "val": 88581, "test": 88581}  # team plan: 70 / 15 / 15 by TransactionDT
COLUMNS_A = re.compile(r"TransactionAmt|card[1-6]|addr[12]|[PR]_emaildomain|DeviceType|DeviceInfo|id_\d\d|V\d+")  # Table 2
V_COLUMN = re.compile(r"V\d+")                   # column set B = set A without V1-V339
SET_SIZES = {"A": 390, "B": 51}                  # team plan p. 2
GRAPH_NODES = 14239                              # team plan p. 4 and graph_info.json
ENTITY_COLUMNS = ["card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo", "DeviceType"]  # team plan p. 4


def sha256_file(path):
    """SHA-256 of a file, read in 1 MB pieces."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


START_HASHES = {**{name: sha256_file(ROOT / name) for name in CODE}, **SHA256}  # code as it was when this process started


def write_once(path, data):
    """Save bytes to `path` and return their SHA-256. If the file already exists it must be byte-identical:
    a finished output is never overwritten (style guide 2)."""
    digest = hashlib.sha256(data).hexdigest()
    if path.exists():
        assert sha256_file(path) == digest, f"{path.name} exists and differs: refusing to overwrite it"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return digest


def parquet_bytes(df):
    """A DataFrame as Parquet bytes, so it can go through write_once()."""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def json_bytes(obj):
    """Readable JSON bytes with a fixed layout, so the same content always gives the same hash."""
    return (json.dumps(obj, indent=2) + "\n").encode()


def load_split(columns):
    """The shared split file, only TransactionID, TransactionDT, isFraud, split and `columns` (pyarrow reads
    just those, so RAM stays low). Checks it is the team's file: same hash, 590,540 payments sorted by
    TransactionDT, train rows first, then validation, then test, in the agreed sizes. Returns the table and
    each split's (first row, end row)."""
    assert sha256_file(SPLIT) == SHA256[SPLIT.name], f"{SPLIT.name} is not the team's file"
    table = pq.read_table(SPLIT, columns=["TransactionID", "TransactionDT", "isFraud", "split", *columns])
    assert table.num_rows == 590540, "split file has the wrong number of rows"
    assert (np.diff(table.column("TransactionDT").to_numpy()) >= 0).all(), "data must be sorted by time"
    split, bounds, start = table.column("split").to_numpy(zero_copy_only=False), {}, 0
    for name in SPLITS:
        bounds[name] = (start, start + SPLIT_SIZES[name])
        assert (split[start:start + SPLIT_SIZES[name]] == name).all(), f"{name} rows are not where the plan puts them"
        start += SPLIT_SIZES[name]
    return table, bounds


def labels(table, bounds):
    """TransactionIDs and isFraud labels of each split, from a table returned by load_split()."""
    ids, y = table.column("TransactionID").to_numpy(), table.column("isFraud").to_numpy()
    return {s: ids[a:b] for s, (a, b) in bounds.items()}, {s: y[a:b] for s, (a, b) in bounds.items()}


def column_set(name):
    """Column set A (paper Table 2: TransactionAmt, card1-6, addr1-2, P/R_emaildomain, DeviceType, DeviceInfo,
    id_01-id_38, V1-V339) or B (A without V1-V339), in the file's column order, as (numeric, categorical).
    Text columns are categorical; every other column is numeric (card1 and addr1 included)."""
    schema = pq.read_schema(SPLIT)
    columns = [f for f in schema if COLUMNS_A.fullmatch(f.name) and (name == "A" or not V_COLUMN.fullmatch(f.name))]
    assert len(columns) == SET_SIZES[name], f"column set {name} should have {SET_SIZES[name]} columns"
    numeric = [f.name for f in columns if "string" not in str(f.type)]
    return numeric, [f.name for f in columns if "string" in str(f.type)]


def numeric_matrix(table, columns, n_train):
    """Paper 3.2 for numeric columns, statistics from the first n_train (training) rows only: missing ->
    training median, then (x - training mean) / training std. A column with one value in training (V107)
    is divided by 1 instead of 0, so it carries no information but keeps set A at the plan's 390 columns."""
    x, constant = np.empty((table.num_rows, len(columns)), np.float32), []
    for j, c in enumerate(columns):
        values = table.column(c).to_pandas().to_numpy(np.float64)   # missing values become NaN
        filled = np.where(np.isnan(values), np.nanmedian(values[:n_train]), values)
        std = filled[:n_train].std()
        constant += [c] if std == 0 else []
        x[:, j] = (filled - filled[:n_train].mean()) / (std if std > 0 else 1.0)
    assert np.isfinite(x).all(), "NaNs left in the features"
    return x, constant


def category_codes(table, columns, n_train):
    """Paper 3.2 for categorical columns, categories from training rows only: 0 = missing, 1 = a value
    never seen in training ("unknown"), 2+ = the training values in sorted order. Returns the codes and
    each column's table size."""
    codes, sizes = np.empty((table.num_rows, len(columns)), np.int64), []
    for j, c in enumerate(columns):
        values = table.column(c).to_pandas()
        categories = sorted(values[:n_train].dropna().unique())
        codes[:, j] = np.where(values.isna(), 0, 2 + pd.Categorical(values, categories=categories).codes)
        sizes.append(len(categories) + 2)
    return codes, sizes


def load_features(name):
    """Column set `name` ("A" or "B") for every split, preprocessed as in paper 3.2 with training statistics
    only. Returns per split (numeric tensor, code tensor), TransactionIDs, labels, category table sizes and
    a short summary. The test rows are scored once, at the end of a run (team plan)."""
    numeric, categorical = column_set(name)
    table, bounds = load_split(numeric + categorical)
    n_train = SPLIT_SIZES["train"]
    x, constant = numeric_matrix(table, numeric, n_train)
    codes, sizes = category_codes(table, categorical, n_train)
    ids, y = labels(table, bounds)
    features = {s: (torch.from_numpy(x[a:b]), torch.from_numpy(codes[a:b])) for s, (a, b) in bounds.items()}
    summary = {"numeric_columns": len(numeric), "categorical_columns": len(categorical),
               "constant_in_training": constant}
    return features, ids, y, sizes, summary


def read_graph_files(shuffle_seed):
    """The files inside graph_files.zip (team plan p. 7): graph_info.json, node_type.npy, the links (the
    real ones, or the shuffled file for `shuffle_seed`) and the three payment node files."""
    assert sha256_file(GRAPH) == SHA256[GRAPH.name], f"{GRAPH.name} is not the team's file"
    edge_file = "graph_edges.npy" if shuffle_seed is None else f"graph_shuf_seed{shuffle_seed}_edges.npy"
    with zipfile.ZipFile(GRAPH) as z:
        read = {name: io.BytesIO(z.read(f"graph_files/{name}"))
                for name in ["graph_info.json", "node_type.npy", edge_file, *(f"payment_nodes_{s}.parquet" for s in SPLITS)]}
    info = json.load(read["graph_info.json"])
    node_type = np.load(read["node_type.npy"], allow_pickle=False)
    edges = np.load(read[edge_file], allow_pickle=False)
    payments = {s: pd.read_parquet(read[f"payment_nodes_{s}.parquet"]) for s in SPLITS}
    return info, node_type, edges, payments


def load_graph(ids, y, shuffle_seed=None):
    """The SOA member's entity graph: links as a [2, E] tensor (each link once per direction), each node's
    entity type, the "unknown" node of each entity column, and each payment's 6 entity node numbers per split.
    Checks the 14,239 nodes and that the payment files line up with the split file row by row."""
    info, node_type, edges, payments = read_graph_files(shuffle_seed)
    assert info["num_nodes"] == len(node_type) == GRAPH_NODES and info["entity_columns"] == ENTITY_COLUMNS
    assert edges.shape[0] == 2 and 0 <= edges.min() and edges.max() < GRAPH_NODES, "links point outside the graph"
    for s in SPLITS:
        assert (payments[s].TransactionID.values == ids[s]).all(), f"graph payment file {s} not aligned"
        assert (payments[s].isFraud.values == y[s]).all(), f"graph payment file {s} has other labels"
    node_ids = {s: torch.as_tensor(payments[s][ENTITY_COLUMNS].to_numpy(np.int64)) for s in SPLITS}
    unknown = torch.as_tensor([info["unknown_node"][c] for c in ENTITY_COLUMNS])
    return torch.as_tensor(edges), torch.as_tensor(node_type), unknown, node_ids


def evaluate(y, score):
    """ROC-AUC and PR-AUC (average precision) of raw scores. Stops on a non-finite score or one-class labels."""
    assert np.isfinite(score).all(), "a score is NaN or infinite"
    assert 0 < y.sum() < len(y), "labels must contain both classes"
    return {"roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score))}


def assert_new_run(run):
    """Stop before any training if this run already has results: a finished run is never overwritten."""
    done = pd.read_csv(RUNS_CSV).run.tolist() if RUNS_CSV.exists() else []
    assert run not in done and not (OUT / f"{run}_history.json").exists(), f"{run} already has results"


def save_run(run, ids, y, scores, history, best_epoch, config, checkpoint):
    """The only code that writes results (style guide 6): <run>_val.parquet and <run>_test.parquet with
    TransactionID, isFraud and score (the raw output before the sigmoid, a logit), <run>_history.json, the
    checkpoint, and one row of runs.csv. The test scores come from the epoch chosen on validation."""
    val, test = evaluate(y["val"], scores["val"]), evaluate(y["test"], scores["test"])
    hashes = {f"{part}_scores_sha256": write_once(OUT / f"{run}_{part}.parquet", parquet_bytes(pd.DataFrame(
        {"TransactionID": ids[part], "isFraud": y[part], "score": scores[part]}))) for part in ["val", "test"]}
    hashes["checkpoint_sha256"] = write_once(OUT / "checkpoints" / f"{run}.pt", checkpoint)
    minutes = round(sum(epoch["seconds"] for epoch in history) / 60, 1)
    config = {**config, "best_epoch": best_epoch, "minutes": minutes, "val": val, "test": test, **hashes,
              "file_sha256": START_HASHES}
    write_once(OUT / f"{run}_history.json", json_bytes({"config": config, "epochs": history}))
    row = {"run": run, "test_roc_auc": round(test["roc_auc"], 6), "test_pr_auc": round(test["pr_auc"], 6),
           "best_epoch": best_epoch, "best_val_roc_auc": round(val["roc_auc"], 6),
           "best_val_pr_auc": round(val["pr_auc"], 6), "minutes": minutes, "finished": time.strftime("%Y-%m-%d %H:%M")}
    pd.DataFrame([row]).to_csv(RUNS_CSV, mode="a", header=not RUNS_CSV.exists(), index=False, lineterminator="\n")
    print(f"[{run}] test ROC-AUC {test['roc_auc']:.4f} | test PR-AUC {test['pr_auc']:.4f}  -> "
          f"{Path(*RUNS_CSV.parts[-3:]).as_posix()}")


def data_checks():
    """Plain-language checks of the split file and the two column sets, as markdown lines for reports/checks.md."""
    table, bounds = load_split(ENTITY_COLUMNS)
    ids, y = labels(table, bounds)
    lines = [f"- Split file `{SPLIT.name}`: SHA-256 matches the team's file; 590,540 payments sorted by TransactionDT.",
             "", "| split | payments | frauds | fraud rate | days |", "|---|---|---|---|---|"]
    days = table.column("TransactionDT").to_numpy() / 86400
    for s, (a, b) in bounds.items():
        lines.append(f"| {s} | {b - a:,} | {int(y[s].sum()):,} | {y[s].mean():.2%} | {days[a]:.1f}-{days[b - 1]:.1f} |")
    lines.append("")
    for name in ["A", "B"]:
        numeric, categorical = column_set(name)
        lines.append(f"- Column set {name}: {len(numeric) + len(categorical)} columns "
                     f"({len(numeric)} numeric, {len(categorical)} categorical).")
    _, constant = numeric_matrix(load_split(column_set("A")[0])[0], column_set("A")[0], SPLIT_SIZES["train"])
    lines.append(f"- Numeric columns with one value in training (divided by 1, not 0): {', '.join(constant)}.")
    return lines, table, ids, y


def graph_checks(table, ids, y):
    """Checks of the SOA member's graph files against the split file, as markdown lines for reports/checks.md:
    node count, link layout, one real value per node, missing and unknown nodes, and the shuffled files."""
    edges, node_type, unknown, node_ids = load_graph(ids, y)
    info = read_graph_files(None)[0]
    key = edges[0] * GRAPH_NODES + edges[1]
    assert len(key.unique()) == edges.shape[1] and torch.isin(edges[1] * GRAPH_NODES + edges[0], key).all()
    assert (edges[0] != edges[1]).all() and (node_type[edges[0]] != node_type[edges[1]]).all()
    assert info["links_before_repeats_removed"] == SPLIT_SIZES["train"] * 6 * 5, "not 6-entity cliques per payment"
    assert not torch.isin(node_ids["train"], unknown).any(), "a training payment uses an unknown node"
    for k, c in enumerate(ENTITY_COLUMNS):
        raw = table.column(c).to_pandas()[:SPLIT_SIZES["train"]]
        nodes = node_ids["train"][:, k].numpy()
        assert ((nodes == info["missing_node"][c]) == raw.isna().values).all(), f"{c}: missing node != missing value"
        assert pd.DataFrame({"n": nodes, "v": raw.astype(str)}).groupby("n").v.nunique().max() == 1, f"{c}: mixed node"
    shuffled = {seed: load_graph(ids, y, seed)[0].shape[1] for seed in [42, 1, 2]}
    unseen = {s: f"{torch.isin(node_ids[s], unknown).any(1).float().mean().item():.2%}" for s in ["val", "test"]}
    return [f"- Graph file `{GRAPH.name}`: SHA-256 matches the team's file; {GRAPH_NODES:,} nodes, as graph_info.json says.",
            f"- Real links: {edges.shape[1]:,} (each link once per direction, no self-links, never two entities "
            f"of the same type); {info['links_before_repeats_removed']:,} before repeats removed = 413,378 x 30.",
            "- Payment node files line up with the split file row by row (TransactionID and isFraud).",
            "- Training payments: every node holds one real value; the 'missing' node is used exactly when the "
            "value is missing; no training payment uses an 'unknown' node.",
            f"- Payments with at least one unknown entity: validation {unseen['val']}, test {unseen['test']}.",
            f"- Shuffled link files (seed: links): {shuffled}; the real graph has {edges.shape[1]:,}."]
