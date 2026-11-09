"""Late fusion of a tabular model and a graph model, with the team plan's checks.

How to run (inside the shared results folder, or point --dir at it):
    python fusion.py rgcn
    python fusion.py rgcn --dir path/to/results
The argument is the graph model's name as used in its file names: sage, gat, rgcn or han.
Needs only numpy, pandas, pyarrow and scikit-learn. Takes a few minutes.

What it does, for every tabular model whose files are in the folder (plain, ft, tabnet), column set
(A, B) and seed (42, 1, 2), once with the real graph and once with the shuffled graph:
  score = z(tabular) + alpha * z(graph)
    z      each model's score standardised with the mean and std of its own validation scores
    alpha  0, 0.025, ..., 1: the one with the best validation PR-AUC (alpha = 0 is the tabular
           model alone; on a tie the smaller alpha is kept)
    seeds are paired: tabular seed 42 with graph seed 42 and shuffled-graph seed 42, and so on.
Then, on the test set (alpha is already fixed, so the test set is used once):
  ROC-AUC and PR-AUC (average precision), and the review budget: the share of frauds caught when
  the top 1% and the top 5% of test payments by score are reviewed.
  95% intervals from a paired bootstrap of the test payments (2,000 resamples, the same resamples
  for every tabular model, column set, seed and graph):
    gain              fused score - tabular model alone
    real - shuffled   fused with the real graph - fused with the shuffled graph

The plan's three checks, per tabular model, column set and metric (summary file). A graph counts as
helping only if all three hold:
  1. the 95% interval of the gain (averaged over the seeds, inside each resample) lies above 0
  2. the mean gain is larger than the spread between the seeds (highest - lowest seed) of the
     tabular model alone
  3. the 95% interval of real - shuffled (averaged over the seeds) lies above 0

Files read (the plan's format: TransactionID, isFraud, score):
  <tabular>_<A|B>_seed<N>_val.parquet / _test.parquet    e.g. plain_A_seed42_test.parquet
  <graph>_seed<N>_val.parquet / _test.parquet             e.g. rgcn_seed42_val.parquet
  <graph>_shuf_seed<N>_val.parquet / _test.parquet        e.g. rgcn_shuf_seed42_test.parquet
Missing files are skipped and listed at the end, so it can simply be run again when more tabular
files arrive.
Files written (in the same folder), named after the graph model and the tabular model(s) fused:
  fusion_<graph>_<tabular>.csv          one row per tabular model, column set, seed and graph
                                        (real / shuffled), e.g. fusion_han_plain.csv
  fusion_<graph>_<tabular>_summary.csv  one row per tabular model, column set and metric: seed mean
                                        and range, intervals and the three checks
  If the folder holds several tabular models, all their names are used, e.g. fusion_han_plain_tabnet.csv.
Every member runs this file unchanged, so all graph models are fused the same way.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TABULAR = ["plain", "ft", "tabnet"]             # tabular models, as named in their files
GRAPHS = ["sage", "gat", "rgcn", "han"]         # graph models, as named in their files
SETS = ["A", "B"]                               # column sets
SEEDS = [42, 1, 2]
ALPHAS = np.round(np.arange(0, 1 + 1e-9, 0.025), 3)
N_BOOT = 2000                                   # bootstrap resamples
BOOT_SEED = 0                                   # fixed, so every member gets the same resamples
CHUNK = 25                                      # resamples computed at a time (memory)
BUDGETS = (0.01, 0.05)                          # review budget: top 1% and top 5% of test payments
METRICS = {"roc": "ROC-AUC", "pr": "PR-AUC"}


# ---- loading ------------------------------------------------------------------------------------
class Folder:
    """Reads score files and checks that every file holds the same payments."""

    def __init__(self, path):
        self.path = Path(path)
        self.ref = {}          # part -> (TransactionID, isFraud) of the first file read
        self.missing = []
        self.prob_warned = False

    def load(self, run, part):
        f = self.path / f"{run}_{part}.parquet"
        if not f.exists():
            self.missing.append(f.name)
            return None
        d = pd.read_parquet(f)
        col = "score" if "score" in d.columns else "logit" if "logit" in d.columns else None
        if col is None or not {"TransactionID", "isFraud"} <= set(d.columns):
            sys.exit(f"{f.name}: needs the columns TransactionID, isFraud and score; it has {list(d.columns)}")
        d = d.sort_values("TransactionID").reset_index(drop=True)
        ids, y, s = d["TransactionID"].to_numpy(), d["isFraud"].to_numpy().astype(int), d[col].to_numpy("float64")
        if d["TransactionID"].duplicated().any():
            sys.exit(f"{f.name}: a TransactionID appears more than once")
        if not np.isfinite(s).all():
            sys.exit(f"{f.name}: the score contains NaN or inf")
        if part not in self.ref:
            self.ref[part] = (ids, y)
        elif not np.array_equal(ids, self.ref[part][0]):
            sys.exit(f"{f.name}: not the same {part} payments as the other files "
                     f"({len(ids):,} rows vs {len(self.ref[part][0]):,}); everyone must use the shared split file")
        elif not np.array_equal(y, self.ref[part][1]):
            sys.exit(f"{f.name}: isFraud differs from the other files for the same TransactionID")
        if not self.prob_warned and s.min() >= 0 and s.max() <= 1:
            print(f"Note: {f.name} has scores between 0 and 1. The plan asks for the output before the "
                  "sigmoid (logit); please check this file holds logits.")
            self.prob_warned = True
        return s

    def pair(self, run):
        """Validation and test scores of one run, or None if either file is missing."""
        v, t = self.load(run, "val"), self.load(run, "test")
        return None if v is None or t is None else (v, t)


# ---- metrics ------------------------------------------------------------------------------------
def zscore(val, test):
    mu, sd = val.mean(), val.std()
    if sd == 0:
        sys.exit("a model gives the same score to every validation payment")
    return (val - mu) / sd, (test - mu) / sd


def pick_alpha(y_val, tab_val, graph_val):
    """alpha with the best validation PR-AUC (the first, i.e. smallest, one on a tie)."""
    best_a, best = 0.0, -1.0
    for a in ALPHAS:
        m = average_precision_score(y_val, tab_val + a * graph_val)
        if m > best:
            best_a, best = float(a), m
    return best_a, best


def caught(y, s, frac):
    """Share of all frauds that are in the top `frac` of payments by score."""
    k = int(np.ceil(frac * len(s)))
    top = np.argsort(-s, kind="mergesort")[:k]
    return y[top].sum() / y.sum()


class Ranked:
    """One test score vector, sorted once, so that ROC-AUC and PR-AUC can be computed quickly for
    many bootstrap resamples. A resample is a vector of counts (how often each payment was drawn);
    the results equal scikit-learn's roc_auc_score / average_precision_score with those counts as
    sample weights (tied scores handled the same way)."""

    def __init__(self, score, y):
        self.order = np.argsort(-score, kind="mergesort")
        s = score[self.order]
        self.ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]   # last payment of each score value
        self.y = y[self.order].astype(np.float32)

    def metrics(self, W):
        # counts are whole numbers far below 2^24, so float32 sums are exact (and faster)
        w = W[:, self.order]
        tp = np.cumsum(w * self.y, axis=1)[:, self.ends].astype(np.float64)
        fp = np.cumsum(w, axis=1)[:, self.ends].astype(np.float64) - tp
        P, N = tp[:, -1:], fp[:, -1:]
        z = np.zeros((len(W), 1))
        tpr, fpr = np.hstack([z, tp / P]), np.hstack([z, fp / N])
        roc = (np.diff(fpr, axis=1) * (tpr[:, 1:] + tpr[:, :-1]) / 2).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        pr = (np.diff(np.hstack([z, tp]), axis=1) * prec).sum(1) / P[:, 0]
        return {"roc": roc, "pr": pr}


def bootstrap(vectors, y, n_boot=N_BOOT):
    """Paired bootstrap: the same resamples for every vector. Returns {name: {"roc": [n_boot], "pr": ...}}."""
    rng = np.random.default_rng(BOOT_SEED)
    ranked = {k: Ranked(v, y) for k, v in vectors.items()}
    out = {k: {m: [] for m in METRICS} for k in vectors}
    n = len(y)
    for start in range(0, n_boot, CHUNK):
        b = min(CHUNK, n_boot - start)
        W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(b)]).astype(np.float32)
        for k, r in ranked.items():
            for m, v in r.metrics(W).items():
                out[k][m].append(v)
    return {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in out.items()}


def interval(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def point(y, s):
    return {"roc": roc_auc_score(y, s), "pr": average_precision_score(y, s)}


# ---- fusion -------------------------------------------------------------------------------------
def fuse_one(folder, tab, cset, graph, n_boot):
    """All seeds of one tabular model and column set with one graph model. Returns detail rows and
    summary rows (empty lists if no seed has all the needed files)."""
    seeds = []
    for seed in SEEDS:
        t = folder.pair(f"{tab}_{cset}_seed{seed}")
        g = folder.pair(f"{graph}_seed{seed}")
        if t is None or g is None:
            continue
        s = folder.pair(f"{graph}_shuf_seed{seed}")
        seeds.append((seed, t, g, s))
    if not seeds:
        return [], []
    y_val, y_test = folder.ref["val"][1], folder.ref["test"][1]

    vectors, info = {}, []
    for seed, t, g, s in seeds:
        tv, tt = zscore(*t)
        e = {"seed": seed, "graph_alone": {"real": point(y_test, g[1])}}
        vectors[(seed, "tab")] = tt
        e["val_tab"] = average_precision_score(y_val, tv)
        for kind, gs in (("real", g), ("shuf", s)):
            if gs is None:
                continue
            gv, gt = zscore(*gs)
            a, val_fused = pick_alpha(y_val, tv, gv)
            vectors[(seed, kind)] = tt + a * gt
            e[kind] = {"alpha": a, "val_fused": val_fused}
            e["graph_alone"][kind] = point(y_test, gs[1])
        info.append(e)
    print(f"  {tab}_{cset} + {graph}: seeds {[e['seed'] for e in info]}, "
          f"bootstrap {n_boot} resamples ...", flush=True)
    boot = bootstrap(vectors, y_test, n_boot)
    pts = {k: point(y_test, v) for k, v in vectors.items()}
    top = {k: {f: caught(y_test, v, f) for f in BUDGETS} for k, v in vectors.items()}

    rows = []
    for e in info:
        seed = e["seed"]
        for kind in ("real", "shuf"):
            if kind not in e:
                continue
            r = {"tabular": tab, "column_set": cset, "seed": seed,
                 "graph": graph if kind == "real" else f"{graph}_shuf", "alpha": e[kind]["alpha"],
                 "val_pr_auc_tabular": e["val_tab"], "val_pr_auc_fused": e[kind]["val_fused"]}
            for m in METRICS:
                gain = boot[(seed, kind)][m] - boot[(seed, "tab")][m]
                r[f"test_{m}_graph_alone"] = e["graph_alone"][kind][m]
                r[f"test_{m}_tabular"] = pts[(seed, "tab")][m]
                r[f"test_{m}_fused"] = pts[(seed, kind)][m]
                r[f"{m}_gain"] = pts[(seed, kind)][m] - pts[(seed, "tab")][m]
                r[f"{m}_gain_lo"], r[f"{m}_gain_hi"] = interval(gain)
                if kind == "real" and "shuf" in e:
                    d = boot[(seed, "real")][m] - boot[(seed, "shuf")][m]
                    r[f"{m}_real_minus_shuf"] = pts[(seed, "real")][m] - pts[(seed, "shuf")][m]
                    r[f"{m}_real_minus_shuf_lo"], r[f"{m}_real_minus_shuf_hi"] = interval(d)
            for f in BUDGETS:
                r[f"top{round(f * 100)}pct_caught_tabular"] = top[(seed, "tab")][f]
                r[f"top{round(f * 100)}pct_caught_fused"] = top[(seed, kind)][f]
            rows.append(r)

    summary = []
    all_shuf = all("shuf" in e for e in info)
    for m in METRICS:
        tab_s = np.array([pts[(e["seed"], "tab")][m] for e in info])
        real_s = np.array([pts[(e["seed"], "real")][m] for e in info])
        gain_boot = np.mean([boot[(e["seed"], "real")][m] - boot[(e["seed"], "tab")][m] for e in info], axis=0)
        gain_lo, gain_hi = interval(gain_boot)
        gain_mean = float((real_s - tab_s).mean())
        spread = float(tab_s.max() - tab_s.min())
        r = {"tabular": tab, "column_set": cset, "graph": graph, "metric": METRICS[m],
             "seeds": " ".join(str(e["seed"]) for e in info), "all_seeds": len(info) == len(SEEDS),
             "alphas": " ".join(f"{e['real']['alpha']:.3f}" for e in info),
             "tabular_mean": tab_s.mean(), "tabular_min": tab_s.min(), "tabular_max": tab_s.max(),
             "fused_mean": real_s.mean(), "fused_min": real_s.min(), "fused_max": real_s.max(),
             "graph_alone_mean": np.mean([e["graph_alone"]["real"][m] for e in info]),
             "gain_mean": gain_mean, "gain_lo": gain_lo, "gain_hi": gain_hi,
             "check1_interval_above_0": gain_lo > 0,
             "seed_spread_tabular": spread, "check2_gain_above_seed_spread": gain_mean > spread}
        if all_shuf:
            shuf_s = np.array([pts[(e["seed"], "shuf")][m] for e in info])
            d_boot = np.mean([boot[(e["seed"], "real")][m] - boot[(e["seed"], "shuf")][m] for e in info], axis=0)
            d_lo, d_hi = interval(d_boot)
            r.update({"fused_shuf_mean": shuf_s.mean(), "real_minus_shuf_mean": float((real_s - shuf_s).mean()),
                      "real_minus_shuf_lo": d_lo, "real_minus_shuf_hi": d_hi, "check3_real_beats_shuf": d_lo > 0})
            r["graph_helps"] = bool(r["check1_interval_above_0"] and r["check2_gain_above_seed_spread"]
                                    and r["check3_real_beats_shuf"])
        else:
            r["check3_real_beats_shuf"] = "shuffled-graph files missing"
            r["graph_helps"] = "incomplete"
        summary.append(r)
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", choices=GRAPHS, help="graph model, as named in its score files")
    ap.add_argument("--dir", default=".", help="the shared results folder (default: this folder)")
    ap.add_argument("--boot", type=int, default=N_BOOT, help=argparse.SUPPRESS)   # quick tests only
    args = ap.parse_args()
    folder = Folder(args.dir)
    print(f"Fusing {args.graph} with the tabular models in {folder.path.resolve()}")

    rows, summary = [], []
    for tab in TABULAR:
        for cset in SETS:
            r, s = fuse_one(folder, tab, cset, args.graph, args.boot)
            rows += r
            summary += s
    if not rows:
        print("\nNothing to fuse: no tabular model and graph model share a seed with all files present.")
        print("Missing:", ", ".join(sorted(set(folder.missing))))
        sys.exit(1)

    fused = "_".join(t for t in TABULAR if any(r["tabular"] == t for r in rows))   # e.g. "plain"
    out = folder.path / f"fusion_{args.graph}_{fused}.csv"
    out_s = folder.path / f"fusion_{args.graph}_{fused}_summary.csv"
    pd.DataFrame(rows).round(5).to_csv(out, index=False)
    S = pd.DataFrame(summary)
    S.round(5).to_csv(out_s, index=False)

    print("\nTest set, mean over seeds (fused with the real graph; gain = fused - tabular alone, 95% interval):")
    for _, r in S.iterrows():
        shuf = (f" | real-shuf {r['real_minus_shuf_mean']:+.4f} [{r['real_minus_shuf_lo']:+.4f}, "
                f"{r['real_minus_shuf_hi']:+.4f}]") if "real_minus_shuf_mean" in r and pd.notna(r.get("real_minus_shuf_mean")) else ""
        print(f"  {r['tabular']:6s} {r['column_set']}  {r['metric']:7s} tabular {r['tabular_mean']:.4f} -> fused "
              f"{r['fused_mean']:.4f}  gain {r['gain_mean']:+.4f} [{r['gain_lo']:+.4f}, {r['gain_hi']:+.4f}]"
              f"{shuf}  graph helps: {r['graph_helps']}  (seeds {r['seeds']}, alpha {r['alphas']})")
    print(f"\nSaved {out.name} and {out_s.name} in {folder.path.resolve()}")
    skipped = sorted(set(folder.missing))
    if skipped:
        print(f"Skipped, file not in the folder yet ({len(skipped)}): " + ", ".join(skipped))


if __name__ == "__main__":
    main()
