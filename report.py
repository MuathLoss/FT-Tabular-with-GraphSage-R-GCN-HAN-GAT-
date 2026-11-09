"""Tables and figures of the FT-Transformer teammate's 12 runs and of their fusion with the team's graph
models, made only from saved files (used by 01_results.ipynb and 02_fusion.ipynb). Nothing is trained or
chosen here: every run's epoch, and every fusion weight, was chosen on validation before the test set was
scored once. Runs and fusions that are not finished yet are simply left out.

What it shows                                                                  -> function
  every finished run, with its scores recomputed from its saved predictions   -> run_table()
  each model's test ROC-AUC, PR-AUC and review budget: mean and seed range     -> summary_table(), fig_test_scores()
  validation ROC-AUC and PR-AUC after every epoch, the chosen epoch marked      -> fig_training()
  overfitting: training vs validation ROC-AUC after every epoch                 -> fig_overfitting()
  precision-recall and ROC curves on the test set                              -> fig_curves()
  share of frauds caught when a team reviews the top 1% / 5% (team plan p. 11)  -> fig_review_budget()
  set A vs set B, and real vs shuffled links: paired test PR-AUC differences
  with 95% intervals from 2,000 paired resamples of test payments (p. 11)       -> comparisons_table(), fig_comparisons()
  fusion with each graph model (fusion.py's tables, saved by run_fusion.py):
    does the graph help, with the team plan's three checks                     -> fusion_summary_table(), fig_fusion()
    seed by seed: weight alpha, validation, test, review budget                -> fusion_seed_table()
"""
import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from common import OUT, ROOT, RUNS_CSV

FIGURES = ROOT / "results" / "figures"
MODELS = {"ft_A": "FT-Transformer, set A (with V)", "ft_B": "FT-Transformer, set B (no V)",
          "sage": "GraphSAGE, real links", "sage_shuf": "GraphSAGE, shuffled links"}
COLORS = {"ft_A": "#2a78d6", "ft_B": "#eb6834", "sage": "#1baf7a", "sage_shuf": "#55544f"}  # colour-blind checked
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
PAPER = {"TabNet": (0.759, 0.149), "TabNet-GraphSAGE": (0.791, 0.206), "LightGBM": (0.834, 0.394)}  # Ahmad et al. 2026 (team plan p. 1): (ROC-AUC, PR-AUC)
COMPARISONS = [("Set A minus set B: what V1-V339 add to the FT-Transformer", "ft_A", "ft_B"),
               ("Real minus shuffled links: are GraphSAGE's links real information?", "sage", "sage_shuf")]
SEEDS = [42, 1, 2]                           # team plan order
RESAMPLES, BOOTSTRAP_SEED = 2000, 20261109   # team plan p. 11: 2,000 paired resamples
SHARES = [0.01, 0.05]                        # team plan p. 11: review the top 1% and 5% of payments
FUSION = ROOT / "results" / "fusion"
GRAPHS = {"sage": "GraphSAGE", "rgcn": "R-GCN", "han": "HAN", "gat": "GAT"}   # the team's four graph models
SETS = {"A": "set A (with V)", "B": "set B (no V)"}
METRICS = {"pr": "PR-AUC", "roc": "ROC-AUC"}  # fusion.py's short names
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def r3(x):
    """Every number is shown with 3 decimals, rounded the same way everywhere."""
    return f"{x:.3f}"


def r4(x):
    """Fusion gains are small, so differences are shown with a sign and 4 decimals."""
    return f"{x:+.4f}"


def tick(ok):
    return "✓" if ok else "✗"


def model_of(run):
    """'sage_shuf_seed42' -> 'sage_shuf'."""
    return re.sub(r"_seed\d+$", "", run)


def finished():
    """runs.csv as a table (one row per finished run), in the plan's model order, or an empty table."""
    if not RUNS_CSV.exists():
        return pd.DataFrame(columns=["run", "model", "seed"])
    runs = pd.read_csv(RUNS_CSV)
    runs["model"], runs["seed"] = runs.run.map(model_of), runs.run.str.extract(r"seed(\d+)$")[0].astype(int)
    order = {"model": list(MODELS), "seed": SEEDS}
    return runs.sort_values(["model", "seed"], key=lambda c: c.map(order[c.name].index))


def scores(run, part):
    """Saved scores of one run ("val" or "test"): TransactionID, isFraud, score (a logit)."""
    return pd.read_parquet(OUT / f"{run}_{part}.parquet")


def capture(y, score, share):
    """Share of all frauds found among the `share` highest-scored payments (the team plan's review budget)."""
    top = np.argsort(-score, kind="mergesort")[:math.ceil(share * len(y))]
    return float(y[top].sum() / y.sum())


def run_table():
    """One row per finished run. Every score is recomputed from the saved predictions; 'Matches' says whether
    it equals the number in runs.csv, so nothing was typed in by hand."""
    rows = []
    for row in finished().itertuples():
        val, test = scores(row.run, "val"), scores(row.run, "test")
        values = {"val ROC-AUC": roc_auc_score(val.isFraud, val.score), "val PR-AUC": average_precision_score(val.isFraud, val.score),
                  "test ROC-AUC": roc_auc_score(test.isFraud, test.score), "test PR-AUC": average_precision_score(test.isFraud, test.score)}
        saved = [row.best_val_roc_auc, row.best_val_pr_auc, row.test_roc_auc, row.test_pr_auc]
        matches = all(abs(a - b) < 1e-6 for a, b in zip(values.values(), saved)) and len(test) == 88581
        rows.append({"Model": MODELS[row.model], "Seed": row.seed, "Chosen epoch": row.best_epoch,
                     "Minutes": row.minutes, **{k: r3(v) for k, v in values.items()}, "Matches": "✓" if matches else "✗"})
    return pd.DataFrame(rows) if rows else "No finished runs yet."


def test_metrics():
    """Per finished run: test ROC-AUC, PR-AUC and the share of frauds caught in the top 1% and 5%."""
    rows = []
    for row in finished().itertuples():
        test = scores(row.run, "test")
        y, s = test.isFraud.to_numpy(), test.score.to_numpy()
        rows.append({"model": row.model, "seed": row.seed, "roc_auc": roc_auc_score(y, s),
                     "pr_auc": average_precision_score(y, s), **{f"top {p:.0%}": capture(y, s, p) for p in SHARES}})
    return pd.DataFrame(rows)


def summary_table():
    """Each model on the test set: mean over its finished seeds, with the lowest and highest seed in brackets."""
    metrics = test_metrics()
    if metrics.empty:
        return "No finished runs yet."
    rows = []
    for model, group in metrics.groupby("model", sort=False):
        cell = {c: f"{r3(group[c].mean())} ({r3(group[c].min())}-{r3(group[c].max())})"
                for c in ["roc_auc", "pr_auc", *(f"top {p:.0%}" for p in SHARES)]}
        rows.append({"Model": MODELS[model], "Seeds": len(group), "Test ROC-AUC": cell["roc_auc"],
                     "Test PR-AUC": cell["pr_auc"], **{f"Frauds caught, top {p:.0%}": cell[f"top {p:.0%}"] for p in SHARES}})
    return pd.DataFrame(rows)


def save(fig, name):
    """Saves a figure to results/figures/<name>.png and returns it, so the notebook shows it (once: the figure
    is closed, so it is not drawn a second time at the end of the cell)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def dot_panel(ax, metrics, column, title, references):
    """One panel of fig_test_scores: one row per model, a dot per seed, a bar at the mean, and the reference
    paper's models as dashed lines labelled at the top."""
    models = list(dict.fromkeys(metrics.model))
    for i, model in enumerate(models):
        values = metrics[metrics.model == model][column]
        ax.plot([values.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.scatter(values, [i] * len(values), s=60, color=COLORS[model], edgecolor="white", linewidth=2, zorder=3)
        ax.text(values.max(), i + 0.32, f"  mean {r3(values.mean())}", color=INK, fontsize=9, va="bottom")
    for name, value in references.items():   # labelled in a free band above the first model
        ax.axvline(value, color=MUTED, lw=1, ls="--", zorder=0)
        ax.text(value, -1.0, f" {name}\n paper {r3(value)}", color=MUTED, fontsize=8, va="center")
    ax.set_yticks(range(len(models)), [MODELS[m] for m in models])
    ax.set_ylim(len(models) - 0.4, -1.4)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="x", color=GRID)


def fig_test_scores():
    """Test PR-AUC and ROC-AUC of every model and seed, next to the reference paper's TabNet, TabNet-GraphSAGE
    hybrid and LightGBM. Two separate panels: the two metrics have different scales."""
    metrics = test_metrics()
    if metrics.empty:
        return "No finished runs yet."
    fig, axes = plt.subplots(1, 2, figsize=(12, 1.2 + 0.8 * metrics.model.nunique()), sharey=True)
    dot_panel(axes[0], metrics, "pr_auc", "Test PR-AUC (dot = one seed, bar = mean)", {k: v[1] for k, v in PAPER.items()})
    dot_panel(axes[1], metrics, "roc_auc", "Test ROC-AUC", {k: v[0] for k, v in PAPER.items()})
    return save(fig, "test_scores")


def fig_training():
    """Validation ROC-AUC (top) and PR-AUC (bottom) after every epoch, one panel per model and one line per
    seed; the ring marks the epoch each run kept (best validation ROC-AUC)."""
    runs = finished()
    if runs.empty:
        return "No finished runs yet."
    models = list(dict.fromkeys(runs.model))
    fig, axes = plt.subplots(2, len(models), figsize=(3.6 * len(models), 6), sharey="row", squeeze=False)
    for j, model in enumerate(models):
        for row, style in zip(runs[runs.model == model].itertuples(), ["-", "--", ":"]):
            epochs = pd.DataFrame(json.loads((OUT / f"{row.run}_history.json").read_text())["epochs"])
            for i, metric in enumerate(["val_roc_auc", "val_pr_auc"]):
                ax = axes[i, j]
                ax.plot(epochs.epoch, epochs[metric], color=COLORS[model], lw=2, ls=style, label=f"seed {row.seed}")
                best = epochs[epochs.epoch == row.best_epoch]
                ax.scatter(best.epoch, best[metric], s=70, facecolor="white", edgecolor=COLORS[model], linewidth=2, zorder=3)
                ax.grid(color=GRID)
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axes[0, j].set_title(MODELS[model], loc="left", color=INK, fontsize=10)
        axes[1, j].set_xlabel("Epoch")
        axes[0, j].legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("Validation ROC-AUC")
    axes[1, 0].set_ylabel("Validation PR-AUC")
    return save(fig, "training_curves")


def fig_overfitting():
    """Training (dashed) vs validation (solid) ROC-AUC after every epoch, one panel per model and one pair of
    lines per seed. Training keeps rising while validation stops: the gap is overfitting, and the ring marks
    the epoch each run kept, before it grows. Training scores are measured during each epoch (dropout on)."""
    runs = finished()
    if runs.empty:
        return "No finished runs yet."
    models = list(dict.fromkeys(runs.model))
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.8), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        for row in runs[runs.model == model].itertuples():
            epochs = pd.DataFrame(json.loads((OUT / f"{row.run}_history.json").read_text())["epochs"])
            ax.plot(epochs.epoch, epochs.train_roc_auc, color=COLORS[model], lw=1.5, ls="--", alpha=0.8)
            ax.plot(epochs.epoch, epochs.val_roc_auc, color=COLORS[model], lw=2)
            best = epochs[epochs.epoch == row.best_epoch]
            ax.scatter(best.epoch, best.val_roc_auc, s=70, facecolor="white", edgecolor=COLORS[model], linewidth=2, zorder=3)
        ax.set_title(MODELS[model], loc="left", color=INK, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.grid(color=GRID)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0, 0].set_ylabel("ROC-AUC")
    axes[0, 0].plot([], [], color=MUTED, ls="--", label="training")
    axes[0, 0].plot([], [], color=MUTED, label="validation")
    axes[0, 0].legend(frameon=False, fontsize=8)
    return save(fig, "overfitting")


def fig_curves():
    """Precision-recall (left) and ROC (right) curves on the test set: one colour per model, one thin line per
    seed. The dashed lines are what random scores would give."""
    runs = finished()
    if runs.empty:
        return "No finished runs yet."
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    for row in runs.itertuples():
        test = scores(row.run, "test")
        precision, recall, _ = precision_recall_curve(test.isFraud, test.score)
        fpr, tpr, _ = roc_curve(test.isFraud, test.score)
        label = MODELS[row.model] if row.seed == runs[runs.model == row.model].seed.iloc[0] else None
        left.plot(recall, precision, color=COLORS[row.model], lw=1.5, alpha=0.8, label=label)
        right.plot(fpr, tpr, color=COLORS[row.model], lw=1.5, alpha=0.8, label=label)
    left.axhline(test.isFraud.mean(), color=MUTED, ls="--", lw=1)
    right.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    for ax, (x, y, title) in zip([left, right], [("Recall", "Precision", "Precision-recall, test set"),
                                                ("False positive rate", "True positive rate", "ROC, test set")]):
        ax.set(xlabel=x, ylabel=y, xlim=(0, 1), ylim=(0, 1.02))
        ax.set_title(title, loc="left", color=INK)
        ax.grid(color=GRID)
    left.legend(frameon=False, fontsize=9)
    return save(fig, "test_curves")


def fig_review_budget():
    """Share of all test frauds found if a team reviews the top 1% (left) or 5% (right) of test payments by
    score (team plan p. 11). Bar = mean over seeds, dots = seeds."""
    metrics = test_metrics()
    if metrics.empty:
        return "No finished runs yet."
    models = list(dict.fromkeys(metrics.model))
    fig, axes = plt.subplots(1, 2, figsize=(12, 1.2 + 0.8 * len(models)), sharey=True)
    for ax, share in zip(axes, SHARES):
        column = f"top {share:.0%}"
        right = min(1.0, metrics[column].max() * 1.3 + 0.05)   # room for the labels, never past 100%
        for i, model in enumerate(models):
            values = metrics[metrics.model == model][column]
            ax.barh(i, values.mean(), height=0.6, color=COLORS[model])
            ax.scatter(values, [i] * len(values), s=40, color=INK, zorder=3)
            ax.text(values.max() + 0.03 * right, i, f"{values.mean():.1%}", va="center", color=INK, fontsize=9)
        ax.set_yticks(range(len(models)), [MODELS[m] for m in models])
        ax.set_ylim(len(models) - 0.4, -0.6)
        ax.set_xlim(0, right)
        ax.set_title(f"Frauds caught when reviewing the top {share:.0%} of payments", loc="left", color=INK)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5, 10]))   # whole percentages only
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="x", color=GRID)
    return save(fig, "review_budget")


def sorted_for_ap(y, score):
    """One run's test payments sorted by score (highest first), with the end of each group of tied scores:
    done once per run, so each bootstrap resample only needs sums."""
    order = np.argsort(-score, kind="mergesort")
    return order, y[order].astype(float), np.r_[score[order][1:] != score[order][:-1], True]


def weighted_ap(prepared, weights):
    """Average precision as sklearn computes it (precision times each step in recall), with a weight per
    payment: a bootstrap resample is the same payments, weighted by how often each was drawn."""
    order, y_sorted, ends = prepared
    w = weights[order]
    tp, fp = np.cumsum(w * y_sorted)[ends], np.cumsum(w * (1 - y_sorted))[ends]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    return float(np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * precision))


def paired_resamples(y, left, right):
    """Mean over seeds of (left - right) test PR-AUC, in each of 2,000 resamples of test payments. Every
    comparison and seed uses the same resamples (same random seed), so the differences are paired."""
    rng, n = np.random.default_rng(BOOTSTRAP_SEED), len(y)
    prepared = [(sorted_for_ap(y, a), sorted_for_ap(y, b)) for a, b in zip(left, right)]
    resampled = np.empty(RESAMPLES)
    for k in range(RESAMPLES):
        weights = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        resampled[k] = np.mean([weighted_ap(a, weights) - weighted_ap(b, weights) for a, b in prepared])
    return resampled


def comparisons_table():
    """Per comparison, using the seeds finished on both sides: the test PR-AUC difference per seed, its mean,
    and a 95% interval of the mean from 2,000 paired resamples of test payments. The team plan counts a gain
    only if the interval is above 0 and the gain is larger than the spread between seeds (here: the larger
    seed range, max - min, of the two models)."""
    runs, rows = finished(), []
    for label, left, right in COMPARISONS:
        seeds = [s for s in SEEDS if s in set(runs[runs.model == left].seed) & set(runs[runs.model == right].seed)]
        if not seeds:
            continue
        tests = {(m, s): scores(f"{m}_seed{s}", "test") for m in (left, right) for s in seeds}
        y = tests[left, seeds[0]].isFraud.to_numpy()
        assert all((t.isFraud.to_numpy() == y).all() for t in tests.values()), "scores not aligned"
        ap = {k: average_precision_score(y, t.score) for k, t in tests.items()}
        deltas = [ap[left, s] - ap[right, s] for s in seeds]
        low, high = np.percentile(paired_resamples(y, *([tests[m, s].score.to_numpy() for s in seeds]
                                                        for m in (left, right))), [2.5, 97.5])
        spread = max(np.ptp([ap[m, s] for s in seeds]) for m in (left, right))
        per_seed = dict(zip(seeds, deltas))
        rows.append({"Comparison": label, "Seeds": len(seeds),
                     **{f"Seed {s}": r3(per_seed[s]) if s in per_seed else "not finished" for s in SEEDS},
                     "Mean difference": r3(np.mean(deltas)), "95% interval": f"{r3(low)} to {r3(high)}",
                     "Interval above 0": "yes" if low > 0 else "no", "Seed spread": r3(spread),
                     "Larger than the spread": "yes" if np.mean(deltas) > spread else "no",
                     "_deltas": deltas, "_mean": float(np.mean(deltas)), "_low": low, "_high": high})
    return pd.DataFrame(rows)


def fig_comparisons(table):
    """The comparisons table as a picture: each seed's difference (dots), the mean (bar) and its 95% interval
    (line). Right of the 0 line = the first model is better."""
    if table.empty:
        return "No comparison has both sides finished yet."
    fig, ax = plt.subplots(figsize=(10, 1.4 + 1.1 * len(table)))
    for i, (_, row) in enumerate(table.iterrows()):
        model = next(left for label, left, _ in COMPARISONS if label == row["Comparison"])
        ax.plot([row["_low"], row["_high"]], [i, i], color=INK, lw=2)
        ax.plot([row["_mean"]] * 2, [i - 0.2, i + 0.2], color=INK, lw=3)
        ax.scatter(row["_deltas"], [i] * len(row["_deltas"]), s=60, color=COLORS[model], edgecolor="white",
                   linewidth=2, zorder=3)
        ax.text(max(row["_high"], *row["_deltas"]), i, f"   {r3(row['_mean'])} ({r3(row['_low'])} to {r3(row['_high'])})",
                va="center", color=INK, fontsize=9)   # right of every mark, so it never covers a dot
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(table)), table.Comparison)
    ax.set_ylim(len(table) - 0.5, -0.5)
    ax.set_xlabel("Difference in test PR-AUC")
    ax.grid(axis="x", color=GRID)
    return save(fig, "comparisons")


def fusion_graphs():
    """The graph models whose fusion with the FT-Transformer is finished, in the team's order."""
    return [g for g in GRAPHS if (FUSION / f"fusion_{g}_ft_inputs.json").exists()]


def fusion_results(part):
    """fusion.py's saved tables for every finished graph model, one after the other: part "" is one row per
    column set, seed and graph (real or shuffled); part "_summary" is one row per column set and metric."""
    tables = [pd.read_csv(FUSION / f"fusion_{g}_ft{part}.csv").assign(model=g) for g in fusion_graphs()]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def fusion_summary_table():
    """Does each graph model help on top of the FT-Transformer? One row per graph model, column set and
    metric, on the test set: the FT-Transformer alone and fused (mean over the seeds, lowest and highest seed
    in brackets), the gain and real-minus-shuffled with their 95% intervals, the team plan's three checks
    and the verdict (the graph helps only if all three hold)."""
    summary, detail = fusion_results("_summary"), fusion_results("")
    if summary.empty:
        return "No fusion finished yet."
    short = {name: m for m, name in METRICS.items()}
    rows = []
    for row in summary.itertuples():   # seed values from the detail table: the summary's means are rounded
        seeds = detail[(detail.graph == row.model) & (detail.column_set == row.column_set)]
        alone, fused = seeds[f"test_{short[row.metric]}_tabular"], seeds[f"test_{short[row.metric]}_fused"]
        rows.append({"Graph model": GRAPHS[row.model], "FT-Transformer on": SETS[row.column_set], "Metric": row.metric,
                     "FT alone": f"{r3(alone.mean())} ({r3(alone.min())}-{r3(alone.max())})",
                     "Fused": f"{r3(fused.mean())} ({r3(fused.min())}-{r3(fused.max())})",
                     "Gain": r4(row.gain_mean), "Gain, 95% interval": f"{r4(row.gain_lo)} to {r4(row.gain_hi)}",
                     "1. Interval above 0": tick(row.check1_interval_above_0),
                     "2. Gain > seed spread": f"{tick(row.check2_gain_above_seed_spread)} (spread {row.seed_spread_tabular:.4f})",
                     "Fused, shuffled links": r3(row.fused_shuf_mean), "Real - shuffled": r4(row.real_minus_shuf_mean),
                     "Real - shuffled, 95% interval": f"{r4(row.real_minus_shuf_lo)} to {r4(row.real_minus_shuf_hi)}",
                     "3. Real beats shuffled": tick(row.check3_real_beats_shuf),
                     "Graph helps": "yes" if row.graph_helps else "no", "Alpha (seeds 42, 1, 2)": row.alphas})
    return pd.DataFrame(rows)


def fusion_seed_table():
    """Seed by seed, for every finished graph model and column set: the weight alpha chosen on validation
    PR-AUC (with the real and with the shuffled links), validation PR-AUC before and after fusion, test
    ROC-AUC and PR-AUC of the graph model alone, the FT-Transformer alone, fused, and fused with the shuffled
    links, and the share of test frauds caught in the top 1% and 5% before and after fusion."""
    detail = fusion_results("")
    if detail.empty:
        return "No fusion finished yet."
    real = detail[~detail.graph.str.endswith("_shuf")]
    shuf = detail[detail.graph.str.endswith("_shuf")].set_index(["model", "column_set", "seed"])
    rows = []
    for row in real.itertuples():
        other = shuf.loc[(row.model, row.column_set, row.seed)]
        cells = {"Graph model": GRAPHS[row.model], "FT-Transformer on": SETS[row.column_set], "Seed": row.seed,
                 "Alpha": f"{row.alpha:.3f}", "Alpha, shuffled": f"{other.alpha:.3f}",
                 "Val PR-AUC: FT alone -> fused": f"{r3(row.val_pr_auc_tabular)} -> {r3(row.val_pr_auc_fused)}"}
        for m, name in METRICS.items():
            cells.update({f"Test {name}: graph alone": r3(getattr(row, f"test_{m}_graph_alone")),
                          f"Test {name}: FT alone": r3(getattr(row, f"test_{m}_tabular")),
                          f"Test {name}: fused": r3(getattr(row, f"test_{m}_fused")),
                          f"Test {name}: fused, shuffled links": r3(other[f"test_{m}_fused"])})
        for p in [1, 5]:
            cells[f"Frauds caught, top {p}%: FT alone -> fused"] = (f"{getattr(row, f'top{p}pct_caught_tabular'):.1%} -> "
                                                                     f"{getattr(row, f'top{p}pct_caught_fused'):.1%}")
        rows.append(cells)
    return pd.DataFrame(rows)


def fusion_panel(ax, summary, detail, column_set, m):
    """One panel of fig_fusion: per graph model, the gain over the FT-Transformer alone (blue) and real minus
    shuffled links (grey): dots = seeds, bar = mean, line = 95% interval. Right of 0 = the graph helps."""
    graphs = list(dict.fromkeys(summary.model))
    for i, graph in enumerate(graphs):
        s = summary[(summary.model == graph) & (summary.column_set == column_set) & (summary.metric == METRICS[m])].iloc[0]
        seeds = detail[(detail.graph == graph) & (detail.column_set == column_set)]
        for offset, key, color in [(-0.15, "gain", COLORS["ft_A"]), (0.15, "real_minus_shuf", COLORS["sage_shuf"])]:
            low, high, mean = s[f"{key}_lo"], s[f"{key}_hi"], s[f"{key}_mean"]
            ax.plot([low, high], [i + offset] * 2, color=color, lw=2)
            ax.plot([mean] * 2, [i + offset - 0.1, i + offset + 0.1], color=color, lw=3)
            ax.scatter(seeds[f"{m}_{key}"], [i + offset] * len(seeds), s=30, color=color, edgecolor="white", zorder=3)
            ax.text(max(high, seeds[f"{m}_{key}"].max()), i + offset, f"  {r4(mean)}", va="center", color=INK, fontsize=8)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(graphs)), [GRAPHS[g] for g in graphs])
    ax.set_ylim(len(graphs) - 0.5, -0.5)
    ax.set_title(f"FT-Transformer {SETS[column_set]}: test {METRICS[m]}", loc="left", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID)


def fig_fusion():
    """The fusion results as a picture: one panel per column set (rows) and metric (columns). For each graph
    model, blue = gain of the fused score over the FT-Transformer alone, grey = fused with the real links
    minus fused with the shuffled links, each with its 95% interval. A graph helps only if both are right
    of 0 (and the gain is larger than the seed spread, see the table)."""
    summary, detail = fusion_results("_summary"), fusion_results("")
    if summary.empty:
        return "No fusion finished yet."
    fig, axes = plt.subplots(2, 2, figsize=(12, 2.6 + 1.6 * summary.model.nunique()), sharey=True, layout="constrained")
    for i, column_set in enumerate(SETS):
        for j, m in enumerate(METRICS):
            fusion_panel(axes[i, j], summary, detail, column_set, m)
    axes[0, 0].plot([], [], color=COLORS["ft_A"], lw=2, label="gain over the FT-Transformer alone")
    axes[0, 0].plot([], [], color=COLORS["sage_shuf"], lw=2, label="real minus shuffled links")
    fig.legend(frameon=False, fontsize=9, loc="outside upper center", ncols=2)   # above the panels, never on data
    fig.supxlabel("Difference on the test set (dots = seeds, bar = mean of the seeds, line = 95% interval)",
                  fontsize=9, color=INK)
    return save(fig, "fusion")
