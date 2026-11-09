# FT-Transformer with graph neural networks for fraud detection (IEEE-CIS)

Does a graph of shared cards, addresses, e-mails and devices help a strong tabular model find fraud? This
repository holds one member's part of a five-person team study on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) data, which builds on
Ahmad et al. (2026), *a hybrid TabNet–Graph neural network framework* ([doi](https://doi.org/10.1016/j.jfds.2026.100203)).

This part has two steps:
1. **Done: 12 training runs.** The **FT-Transformer** (Gorishniy et al., 2021), a Transformer for tables, on
   two column sets: with and without the 339 anonymous `V` columns. And **GraphSAGE** (Hamilton et al.,
   2017) on the team's shared entity graph, with the real links and with randomly shuffled links as a
   control. Three seeds each (42, 1, 2).
2. **In progress: fusion.** The FT-Transformer combined with each of the team's four graph models. R-GCN and
   HAN are done; GraphSAGE and GAT come next (see [Fusion](#fusion)).

## Results

Test set: the last 15% of payments in time, scored once per run and never used to choose anything. Mean of
3 seeds, with the lowest and highest seed in brackets.

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| FT-Transformer, all 390 columns (set A, with V) | 0.853 (0.847-0.856) | 0.403 (0.396-0.416) |
| FT-Transformer, 51 columns (set B, without V) | 0.757 (0.751-0.763) | 0.214 (0.202-0.222) |
| GraphSAGE, real links (no payment columns) | 0.759 (0.754-0.762) | 0.139 (0.133-0.142) |
| GraphSAGE, shuffled links | 0.754 (0.747-0.759) | 0.138 (0.131-0.142) |
| *Reference paper: LightGBM* | *0.834* | *0.394* |
| *Reference paper: hybrid TabNet-GraphSAGE* | *0.791* | *0.206* |

PR-AUC is the main score because only 3.5% of payments are fraud (guessing gives a PR-AUC of 0.035).

- The FT-Transformer with all columns is on par with the reference paper's best model (LightGBM), and well
  above its hybrid TabNet-GraphSAGE.
- The V columns nearly double the FT-Transformer's PR-AUC: +0.189 (95% interval 0.175 to 0.203, from
  2,000 paired bootstrap resamples of the test payments).
- GraphSAGE on the real graph scores the same as on shuffled links: +0.001 PR-AUC (95% interval -0.001 to
  0.003, which includes 0). On its own, the graph structure adds nothing measurable beyond each payment's
  own card, address, e-mail and device. Whether it helps *on top of* the FT-Transformer is what the fusion
  step tests.

![Test scores per model, each dot one seed](results/figures/test_scores.png)
![ROC and precision-recall curves on the test set](results/figures/test_curves.png)
![Set A minus set B, and real minus shuffled links, with 95% intervals](results/figures/comparisons.png)

More graphs (training curves, overfitting, frauds caught in the top 1% and 5%, set A vs B and real vs
shuffled with bootstrap intervals) are in [`01_results.ipynb`](01_results.ipynb) and
[`results/figures/`](results/figures/). Every run's score per epoch is in
`results/graph_vs_V/<run>_history.json`, and all 12 runs are in
[`results/graph_vs_V/runs.csv`](results/graph_vs_V/runs.csv).

## How the study is set up

Every member of the team uses the same data split, the same graph and the same rules, so scores from
different people's models can be compared and combined row by row (by TransactionID).

| | FT-Transformer | GraphSAGE |
|---|---|---|
| Data | 590,540 Kaggle training payments, sorted by time: 70% train, 15% validation, 15% test | same |
| Inputs | set A: the reference paper's 390 columns (369 numeric, 21 categorical); set B: the same without V1-V339 (51) | each payment's 6 entity nodes; no payment columns |
| Preprocessing | training medians, means and standard deviations; categories from training, unseen values become "unknown" | none |
| Model | 192-dimensional tokens, 3 pre-norm Transformer blocks, 8 heads, [CLS] head | node + entity-type embeddings (64), 2 mean layers, head 384 -> 128 -> 1 |
| Training | Adam (lr 0.001), batch 2,048, fraud-weighted cross-entropy, up to 80 epochs, stop 10 epochs after the best | same, 12 epochs, 10% of entities swapped for "unknown" in training |
| Epoch kept | best validation ROC-AUC | same |
| Saved | logit for every validation and test payment | same |

The full method, written for the paper, is in [`reports/method.md`](reports/method.md). Every setting and
decision, checked against the team plan, is in [`reports/plan.md`](reports/plan.md).

## What is in this repository

```
├── runs.py                 every training run by name (start here to read the code)
├── models.py               the FT-Transformer and GraphSAGE, one fit() and one predict()
├── common.py               loading the shared files, preprocessing, scoring, saving results
├── report.py               the tables and graphs of 01_results.ipynb and 02_fusion.ipynb
├── fusion.py               the team's shared fusion script, used unchanged
├── run_fusion.py           runs fusion.py for one graph model and saves its results (python run_fusion.py rgcn)
├── 00_train_runs.ipynb     Colab notebook that ran the 12 runs (with its output)
├── 01_results.ipynb        every table and graph of the 12 runs (with its output)
├── 02_fusion.ipynb         every table and graph of the fusion, and a re-run that proves it (with its output)
├── reports/
│   ├── method.md           the method paragraphs for the paper
│   ├── plan.md             settings, decisions and the check against the team plan
│   └── checks.md           checks of the shared data and graph files
├── results/
│   ├── graph_vs_V/         runs.csv, each run's history and training log, checkpoints/ (trained models)
│   ├── figures/            the graphs
│   ├── from_teammates/     R-GCN, HAN and GAT score files (in Drive only, see below)
│   └── fusion/             fusion.py's result tables per graph model, and a record of their inputs
├── data/README.md          where the data comes from (not included)
└── references/README.md    papers to cite
```

## Reproducing the runs

1. **Data.** Accept the competition rules on Kaggle and download the data. The runs need the team's two
   shared input files, built from it (a sorted 70/15/15 split and the entity graph, see
   [`data/README.md`](data/README.md)). They cannot be published, and the code checks their SHA-256 hashes,
   so a run only starts with the exact files the team used.
2. **Environment.** Python with PyTorch, NumPy, pandas, pyarrow, scikit-learn and matplotlib
   (`requirements.txt`). The results come from Google Colab on one NVIDIA A100 (Python 3.13, PyTorch
   2.14.1, CUDA 13.0). `requirements.lock.txt` is the exact local environment used for the checks.
3. **Run.**
   - In Google Colab: put this folder in Google Drive as `My Drive/IEEE Fraud Detection`, open
     `00_train_runs.ipynb`, choose a GPU (A100 or L4) and run all cells. Then open `01_results.ipynb` (no
     GPU needed) for the tables and graphs.
   - On a PC: `python runs.py checks`, `python runs.py smoke`, then `python runs.py all`.

   `checks` and `smoke` train nothing. `all` runs the 12 runs and skips finished ones; a finished result is
   never overwritten.
4. **Check.** The runs use deterministic GPU kernels and full 32-bit precision, and two identical runs give
   identical scores. Each history file stores the SHA-256 of the code, the input files, the score files and
   the checkpoint. Section A of `01_results.ipynb` recomputes every score from the saved files and marks it
   ✓ when it matches `runs.csv`.

The trained models are in `results/graph_vs_V/checkpoints/<run>.pt`. Each is a PyTorch `state_dict` for
`models.FTTransformer` or `models.GraphSAGE`, built as in `runs.py`. GraphSAGE also needs the shared graph
file.

## Fusion

The FT-Transformer is combined with each graph model of the team, using the team's shared
[`fusion.py`](fusion.py) unchanged, so that every graph model is fused the same way:

- score = z(FT-Transformer) + alpha × z(graph), each model standardised on its own validation scores;
- alpha from 0 to 1 in steps of 0.025, chosen on validation PR-AUC (alpha = 0 is the FT-Transformer alone);
- seeds are paired (FT seed 42 with graph seed 42, and so on), and the same is done with the graph model
  trained on shuffled links;
- the test set is scored once, with 2,000 paired bootstrap resamples. A graph counts as helping only if
  (1) the 95% interval of the gain is above 0, (2) the gain is larger than the spread between the
  FT-Transformer's seeds, and (3) the real links beat the shuffled links.

[`run_fusion.py`](run_fusion.py) runs `fusion.py` for one graph model (it needs all 24 score files in one
folder), checks that its FT-Transformer-alone scores equal `runs.csv`, and saves its tables to
[`results/fusion/`](results/fusion/) with the SHA-256 of every input. [`02_fusion.ipynb`](02_fusion.ipynb)
shows every metric (per seed: alpha, validation, test, graph alone, shuffled links, review budget) and runs
the fusion again to show that it gives the same results.

**Results so far** (test set, mean of 3 seeds; gain = fused minus the FT-Transformer alone, with its 95%
interval):

| Graph model | FT-Transformer on | Test PR-AUC: alone -> fused | Gain | Graph helps? | Test ROC-AUC: alone -> fused | Gain | Graph helps? |
|---|---|---|---|---|---|---|---|
| R-GCN | set A (with V) | 0.403 -> 0.415 | +0.0115 (+0.0086 to +0.0144) | no (checks 2, 3 fail) | 0.853 -> 0.865 | +0.0124 (+0.0100 to +0.0148) | **yes** |
| R-GCN | set B (no V) | 0.214 -> 0.216 | +0.0018 (-0.0012 to +0.0048) | no (checks 1, 2, 3 fail) | 0.757 -> 0.787 | +0.0305 (+0.0258 to +0.0353) | **yes** |
| HAN | set A (with V) | 0.403 -> 0.416 | +0.0125 (+0.0100 to +0.0152) | no (check 2 fails) | 0.853 -> 0.864 | +0.0113 (+0.0094 to +0.0132) | **yes** |
| HAN | set B (no V) | 0.214 -> 0.217 | +0.0028 (+0.0003 to +0.0055) | no (checks 2, 3 fail) | 0.757 -> 0.788 | +0.0311 (+0.0272 to +0.0351) | **yes** |
| GraphSAGE | | to do | | | | | |
| GAT | | coming | | | | | |

- **ROC-AUC (ranking all payments): both graph models help**, with and without the V columns, and pass all
  three checks. Their real links beat the shuffled ones, but by much less than the total gain (for R-GCN
  on set B: +0.002 of +0.031). So most of the gain comes from what the graph model learned about each
  card, address, e-mail and device, not from the links between them.
- **PR-AUC (the main metric, because fraud is rare): neither helps by the team's rules.** With the V columns
  the gain is real (about +0.012, interval above 0), but smaller than the spread between the
  FT-Transformer's own seeds (0.020). HAN's links do beat its shuffled links on PR-AUC (+0.003, interval
  +0.002 to +0.004), the only case so far. Without the V columns the PR-AUC gain is close to 0.

The teammates' score files are not published here, because they contain the Kaggle labels.

![Fusion: gain over the FT-Transformer alone and real minus shuffled links, with 95% intervals](results/figures/fusion.png)

## Data and licences

- **Code:** MIT licence ([`LICENSE`](LICENSE)). `fusion.py` was written by the team for shared use and is
  included with the team's agreement.
- **Data:** IEEE-CIS Fraud Detection, Kaggle. Under the competition rules it may be used for non-commercial
  purposes only (including academic research and education) and may not be redistributed. So it is not
  included, and neither is any file that contains it (the split, the graph files and the score files).
- **Trained models and results** (`results/`): these come from training on the Kaggle data. They are shared
  for non-commercial research and education, in line with the data's terms.

## References

- Ahmad, H., Bhuiyan, I., Bhuiyan, T. and Hoque, A. (2026). Scalable and interpretable fraud detection in
  FinTech: Evidence from a hybrid TabNet–Graph neural network framework. *The Journal of Finance and Data
  Science*, 12, 100203. <https://doi.org/10.1016/j.jfds.2026.100203>
- Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A. (2021). Revisiting deep learning models for
  tabular data. *NeurIPS 2021*.
- Hamilton, W. L., Ying, R. and Leskovec, J. (2017). Inductive representation learning on large graphs.
  *NeurIPS 2017*.
