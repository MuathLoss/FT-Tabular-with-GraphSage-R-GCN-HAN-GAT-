"""FT-Transformer and GraphSAGE runs of the FT-Transformer teammate (references/Team_plan_graph_vs_V.pdf, p. 8).

Every member uses the same split file and the same graph files, so a score difference comes from the model
only. Nothing in models.py is changed here: each run calls it with one thing changed.

How to run (PyCharm terminal or the Colab notebook, inside the project folder, one name at a time):
    python runs.py checks         check the split file, column sets and graph files; trains nothing
    python runs.py smoke          time a few training steps of each model on this GPU; saves nothing
    python runs.py ft_A_seed42    one run
    python runs.py all            every run below in order, skipping finished ones (safe to restart)
Each run adds one line to results/graph_vs_V/runs.csv and saves the team's three hand-in files:
<run>_val.parquet and <run>_test.parquet (TransactionID, isFraud, score = logit) and <run>_history.json.

Run               What changes                                                   Time on the RTX 3050 (smoke)
sage_seed42       GraphSAGE on the SOA member's entity graph, no payment columns ~2 min (12 epochs x 9 s)
sage_shuf_seed42  same model on the SOA member's shuffled links for seed 42      ~2 min
ft_B_seed42       FT-Transformer, column set B (51 columns, no V1-V339)           ~1.5 min per epoch
ft_A_seed42       same, column set A (paper Table 2, 390 columns, with V)        ~23 min per epoch
The FT-Transformer runs until 10 epochs after its best validation ROC-AUC (at least 11 epochs, at most 80).
On a Colab A100 or L4, `python runs.py smoke` prints that GPU's times.
Every run also exists with seed1 and seed2 (same run, different random seed): 12 runs in all.
GraphSAGE never sees payment columns, so Reset fuses each GraphSAGE run with both column sets; fusion and
evaluation are Reset's part of the plan (no training). The test set is scored once per run, at its end,
with the epoch chosen on validation ROC-AUC.
"""
import argparse
import json
import math
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from common import (OUT, ROOT, RUNS_CSV, SPLIT_SIZES, START_HASHES, assert_new_run, data_checks, graph_checks,
                    labels, load_features, load_graph, load_split, save_run)
from models import (DEVICE, FTTransformer, GraphSAGE, checkpoint_bytes, fit, predict, set_seed, train_epoch,
                    training_setup)

SEEDS = [42, 1, 2]       # team plan: seeds 42, 1 and 2 for every run
BATCH = 2048             # payments per optimizer step (our choice; the plan leaves it free)
LR = 1e-3                # Adam learning rate (our choice; the plan leaves it free)
FT_EPOCHS = 80           # paper Table 3: tabular models up to 80 epochs
FT_PATIENCE = 10         # team rule: stop 10 epochs after the best validation ROC-AUC
SAGE_EPOCHS = 12         # paper Table 3 and team plan p. 5: 12 epochs, keep the best validation ROC-AUC
GPU_SHARE = 0.8          # PyTorch may use 80% of the GPU's memory, so Windows never spills it into RAM
SMOKE_STEPS = 10         # training steps timed per model by `smoke`
ORDER = ([f"sage_seed{s}" for s in SEEDS] + [f"sage_shuf_seed{s}" for s in SEEDS]
         + [f"ft_B_seed{s}" for s in SEEDS] + [f"ft_A_seed{s}" for s in SEEDS])  # `all`: shortest runs first


def environment():
    """The GPU and software a run used, written into its history file (part of the progress fingerprint)."""
    assert torch.cuda.is_available(), "a GPU is required"
    torch.cuda.set_per_process_memory_fraction(GPU_SHARE)
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version()}


def ft_pieces(column_set):
    """How many GPU steps each 2,048-payment FT-Transformer batch is split into (same maths for any value;
    written in the history file). Set B (52 tokens) fits in one step. Set A's attention over 391 tokens needs
    about 4 GB per 256 payments in full 32-bit precision, so it depends on the GPU's memory."""
    memory = torch.cuda.get_device_properties(0).total_memory / 2**30
    return 1 if column_set == "B" else 2 if memory >= 39 else 4 if memory >= 20 else 8


def train(run, model, data, ids, y, settings, epochs, patience=None, pieces=1, eval_batch=16384):
    """The shared part of every run: fit on train, keep the epoch with the best validation ROC-AUC, then score
    validation and test once with that epoch and save the hand-in files. An interrupted run continues from
    results/graph_vs_V/progress/<run>.pt, but only with the same code, data, settings and GPU type."""
    progress = OUT / "progress" / f"{run}.pt"
    settings = {**settings, **environment()}
    fingerprint = json.dumps({"settings": settings, "files": START_HASHES}, sort_keys=True)
    print(f"[{run}] {settings['model']} on {settings['gpu']}", flush=True)
    history, best_epoch = fit(model, data, y, epochs, settings["seed"], BATCH, LR, progress, fingerprint,
                              patience, pieces, eval_batch)
    scores = {part: predict(model, data[part], eval_batch) for part in ["val", "test"]}  # the test set's only use
    save_run(run, ids, y, scores, history, best_epoch, settings, checkpoint_bytes(model))
    progress.unlink()


def run_ft(run, column_set, seed):
    """FT-Transformer on column set A or B (team plan p. 8): up to 80 epochs (paper Table 3), stopping 10 epochs
    after the best validation ROC-AUC (team rule)."""
    assert_new_run(run)
    set_seed(seed)
    features, ids, y, sizes, summary = load_features(column_set)
    model = FTTransformer(features["train"][0].shape[1], sizes).to(DEVICE)
    pieces = ft_pieces(column_set)
    settings = {"model": "FT-Transformer", "columns": column_set, "seed": seed, "max_epochs": FT_EPOCHS,
                "patience": FT_PATIENCE, "selected_on": "val_roc_auc", "batch": BATCH, "pieces_per_batch": pieces,
                "lr": LR, "dim": 192, "blocks": 3, "heads": 8, "hidden": 384, "dropout": 0.1, "grad_clip": 1.0, **summary}
    train(run, model, features, ids, y, settings, FT_EPOCHS, FT_PATIENCE, pieces, eval_batch=4096 // pieces)


def run_sage(run, seed, shuffled):
    """GraphSAGE on the SOA member's entity graph (team plan p. 4-5, 8): 12 epochs, keeping the best validation
    ROC-AUC. shuffled=True uses the SOA member's shuffled links for this seed (team plan p. 11): every entity
    keeps how many payments use it and every payment keeps its real entities, but which entities are linked is
    random. It is the control for "do the links carry real information?"."""
    assert_new_run(run)
    set_seed(seed)
    ids, y = labels(*load_split([]))
    edges, node_type, unknown, node_ids = load_graph(ids, y, seed if shuffled else None)
    model = GraphSAGE(edges, node_type, unknown).to(DEVICE)
    settings = {"model": "GraphSAGE", "columns": "none", "seed": seed, "graph": "shuffled" if shuffled else "real",
                "links": int(edges.shape[1]), "epochs": SAGE_EPOCHS, "selected_on": "val_roc_auc", "batch": BATCH,
                "lr": LR, "dim": 64, "layers": 2, "dropout": 0.2, "unknown_swap": 0.1, "grad_clip": 1.0}
    train(run, model, {s: (node_ids[s],) for s in node_ids}, ids, y, settings, SAGE_EPOCHS)


def run_checks():
    """Checks the two shared files before any training and writes reports/checks.md. Trains nothing."""
    data_lines, table, ids, y = data_checks()
    text = "\n".join(["# Data and graph checks", "",
                      "Written by `python runs.py checks`. Every line is an `assert` in common.py that passed.", "",
                      "## Split file and column sets", "", *data_lines, "", "## Graph files", "",
                      *graph_checks(table, ids, y), ""])
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "checks.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


def smoke_one(name, make_model, data, y, pieces):
    """Trains SMOKE_STEPS batches twice from the same seed and scores 20,000 validation payments: prints the
    time per batch, an estimate per epoch, the peak GPU memory, and whether the two runs gave identical scores."""
    rows, n_val = torch.arange(SMOKE_STEPS * BATCH), 20000
    y_train = torch.as_tensor(y["train"][:len(rows)], dtype=torch.float32, device=DEVICE)
    scores = []
    for _ in range(2):
        set_seed(42)
        model = make_model()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        train_epoch(model, tuple(t[rows] for t in data["train"]), y_train, *training_setup(model, y_train, LR, 42), BATCH, pieces)
        torch.cuda.synchronize()
        step, middle = (time.time() - start) / SMOKE_STEPS, time.time()
        scores.append(predict(model, tuple(t[:n_val] for t in data["val"]), 4096 // pieces))
        scoring = (time.time() - middle) * SPLIT_SIZES["val"] / n_val
    minutes = (step * math.ceil(SPLIT_SIZES["train"] / BATCH) + scoring) / 60
    print(f"{name}: {step:.2f} s per batch -> about {minutes:.1f} min per epoch | peak GPU "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB | two identical runs: "
          f"{'identical' if np.array_equal(*scores) else 'DIFFERENT'}", flush=True)


def run_smoke():
    """Times each model on this GPU (GraphSAGE, FT-Transformer on B and on A) and checks reproducibility.
    Saves nothing; use it to fill in the run times above and to plan `all`."""
    print(environment(), flush=True)
    ids, y = labels(*load_split([]))
    edges, node_type, unknown, node_ids = load_graph(ids, y)
    smoke_one("GraphSAGE", lambda: GraphSAGE(edges, node_type, unknown).to(DEVICE),
              {s: (node_ids[s],) for s in node_ids}, y, 1)
    for column_set in ["B", "A"]:
        features, _, _, sizes, _ = load_features(column_set)
        smoke_one(f"FT-Transformer {column_set} ({ft_pieces(column_set)} pieces per batch)",
                  lambda: FTTransformer(features["train"][0].shape[1], sizes).to(DEVICE), features, y,
                  ft_pieces(column_set))
        del features


def run_all():
    """Every run in ORDER, each in its own Python process (a clean GPU each time), skipping runs already in
    runs.csv: after a crash or a Colab disconnect, start it again and it continues where it stopped. Each
    run's output is shown and also appended to results/graph_vs_V/logs/<run>.log."""
    for run in ORDER:
        if RUNS_CSV.exists() and run in pd.read_csv(RUNS_CSV).run.tolist():
            print(f"[{run}] already done, skipped", flush=True)
            continue
        log = OUT / "logs" / f"{run}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f, subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "runs.py"), run], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line)
        assert process.returncode == 0, f"{run} failed: see {log}"


RUNS = {
    "checks": run_checks,
    "smoke": run_smoke,
    "all": run_all,
    **{f"ft_{c}_seed{s}": (lambda c=c, s=s: run_ft(f"ft_{c}_seed{s}", c, s)) for c in ["A", "B"] for s in SEEDS},
    **{f"sage_seed{s}": (lambda s=s: run_sage(f"sage_seed{s}", s, shuffled=False)) for s in SEEDS},
    **{f"sage_shuf_seed{s}": (lambda s=s: run_sage(f"sage_shuf_seed{s}", s, shuffled=True)) for s in SEEDS},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", choices=list(RUNS))
    RUNS[parser.parse_args().run]()
