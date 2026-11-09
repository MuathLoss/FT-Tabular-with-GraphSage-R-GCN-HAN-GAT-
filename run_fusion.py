"""Fusion of our FT-Transformer with one graph model, using the team's fusion.py unchanged.

fusion.py reads every score file from one folder and writes its two result tables into that folder. Our
score files live in two places (ours in results/graph_vs_V/, the teammates' in
results/from_teammates/<model>/), so this script:
  1. copies (never moves) the 12 FT-Transformer files (sets A and B, seeds 42, 1, 2, validation and test)
     and the graph model's 12 files (real and shuffled links) into a temporary folder;
  2. runs fusion.py there, unchanged (alpha chosen on validation PR-AUC, test scored once, 2,000 paired
     bootstrap resamples, the team plan's three checks);
  3. checks that its "FT-Transformer alone" test scores equal runs.csv, which proves it used our files;
  4. saves to results/fusion/, never overwriting a finished result:
       fusion_<graph>_ft.csv            one row per column set, seed and graph (real or shuffled)
       fusion_<graph>_ft_summary.csv    one row per column set and metric, with the three checks
       fusion_<graph>_ft_inputs.json    SHA-256 of fusion.py, of every input file and of both tables
recheck() runs it all again in a temporary folder and compares with the saved result (02_fusion.ipynb).

How to run (inside the project folder; CPU only, about 3 minutes per graph model):
    python run_fusion.py rgcn        # or sage, han, gat
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT, ROOT, json_bytes, sha256_file, write_once

FUSION = ROOT / "results" / "fusion"
TEAM = ROOT / "results" / "from_teammates"
GRAPH_FOLDERS = {"sage": OUT, "rgcn": TEAM / "rgcn", "han": TEAM / "han", "gat": TEAM / "gat"}
SEEDS = [42, 1, 2]


def input_files(graph):
    """The 24 score files fusion.py needs for one graph model; stops if any is missing."""
    ft = [OUT / f"ft_{c}_seed{s}_{p}.parquet" for c in "AB" for s in SEEDS for p in ["val", "test"]]
    links = [GRAPH_FOLDERS[graph] / f"{graph}{shuf}_seed{s}_{p}.parquet"
             for shuf in ["", "_shuf"] for s in SEEDS for p in ["val", "test"]]
    missing = [f.name for f in ft + links if not f.exists()]
    assert not missing, f"missing score files: {', '.join(missing)}"
    return ft + links


def run_team_fusion(graph, files, work):
    """Copies the files into `work`, runs fusion.py there unchanged, and returns its two tables' paths and
    what it printed."""
    for f in files:
        shutil.copy2(f, work / f.name)
    done = subprocess.run([sys.executable, str(ROOT / "fusion.py"), graph, "--dir", str(work)],
                          check=True, capture_output=True, text=True)
    return (work / f"fusion_{graph}_ft.csv", work / f"fusion_{graph}_ft_summary.csv"), done.stdout


def check(detail, summary):
    """All 12 rows and 4 summary rows are there, every seed was used, and the FT-Transformer-alone test
    scores equal runs.csv (fusion.py rounds to 5 decimals, runs.csv to 6)."""
    assert len(detail) == 12 and len(summary) == 4, "fusion.py did not fuse every column set, seed and graph"
    assert summary.all_seeds.all(), "a seed is missing"
    runs = pd.read_csv(OUT / "runs.csv").set_index("run")
    for row in detail.itertuples():
        saved = runs.loc[f"ft_{row.column_set}_seed{row.seed}"]
        assert abs(row.test_roc_tabular - saved.test_roc_auc) < 2e-5, f"{row.column_set} seed {row.seed}: ROC-AUC differs"
        assert abs(row.test_pr_tabular - saved.test_pr_auc) < 2e-5, f"{row.column_set} seed {row.seed}: PR-AUC differs"


def save(graph, files, tables):
    """Writes both tables and the input record to results/fusion/ (write-once)."""
    record = {"graph": graph, "finished": time.strftime("%Y-%m-%d %H:%M"), "fusion.py": sha256_file(ROOT / "fusion.py"),
              "inputs": {f.name: sha256_file(f) for f in files}, "outputs": {}}
    for table in tables:
        record["outputs"][table.name] = write_once(FUSION / table.name, table.read_bytes())
    write_once(FUSION / f"fusion_{graph}_ft_inputs.json", json_bytes(record))
    print(f"[fusion {graph}] saved {', '.join(t.name for t in tables)} -> results/fusion/")


def same_table(saved, new):
    """True if two result tables hold the same values (numbers to within 2e-5: fusion.py rounds to 5
    decimals, and another computer may round the last digit the other way)."""
    a, b = pd.read_csv(saved), pd.read_csv(new)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    numbers = a.select_dtypes("number").columns
    return bool(np.allclose(a[numbers], b[numbers], atol=2e-5, equal_nan=True)
                and a.drop(columns=numbers).equals(b.drop(columns=numbers)))


def recheck(graph):
    """Runs the fusion again from the score files and compares it with the saved result: the same fusion.py
    and input files (SHA-256) and the same numbers in both tables. About 3 minutes."""
    record = json.loads((FUSION / f"fusion_{graph}_ft_inputs.json").read_text())
    files = input_files(graph)
    with tempfile.TemporaryDirectory() as folder:
        tables, _ = run_team_fusion(graph, files, Path(folder))
        same = all(same_table(FUSION / t.name, t) for t in tables)
    tick = lambda ok: "✓" if ok else "✗"
    return {"Graph model": graph, "Fused on": record["finished"],
            "Same fusion.py": tick(sha256_file(ROOT / "fusion.py") == record["fusion.py"]),
            "Same 24 score files": tick(all(sha256_file(f) == record["inputs"][f.name] for f in files)),
            "Same results when run again": tick(same)}


def main(graph):
    assert not (FUSION / f"fusion_{graph}_ft_inputs.json").exists(), f"fusion with {graph} is already done"
    files = input_files(graph)
    print(f"[fusion {graph}] running fusion.py on {len(files)} score files (about 3 minutes) ...", flush=True)
    with tempfile.TemporaryDirectory() as folder:
        tables, printed = run_team_fusion(graph, files, Path(folder))
        print(printed.split("\nSaved ")[0])
        check(pd.read_csv(tables[0]), pd.read_csv(tables[1]))
        save(graph, files, tables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", choices=list(GRAPH_FOLDERS))
    main(parser.parse_args().graph)
