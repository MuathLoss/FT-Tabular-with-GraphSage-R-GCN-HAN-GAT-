# Results: what is in this folder

Everything the 12 training runs produced (9 November 2026, Colab A100), the teammates' graph models'
scores, and (next) the fusion results. Nothing in here is edited by hand. Do not rename or move these
files: the code, the results notebook and the fusion script find them by these names.

```
results/
├── README.md        <- this page
├── graph_vs_V/      <- our 12 runs (the folder is named after the team study, "graph vs V")
│   ├── runs.csv                 <- the results table: one row per run
│   ├── <run>_val.parquet        <- the run's score for every validation payment (88,581)
│   ├── <run>_test.parquet       <- the run's score for every test payment (88,581)
│   ├── <run>_history.json       <- every training epoch's scores, plus every setting of the run
│   ├── checkpoints/<run>.pt     <- the trained model itself (its weights)
│   └── logs/<run>.log           <- what the run printed while training
├── figures/         <- the graphs as pictures, saved by 01_results.ipynb
├── from_teammates/  <- R-GCN, HAN and GAT score files, as the teammates sent them (see its README)
│   ├── rgcn/  han/  gat/
└── fusion/          <- next step: the FT-Transformer fused with each graph model (see its README)
```

The `.parquet` files are tables with three columns: TransactionID, isFraud (1 = fraud) and score. They open
in Python (`pandas.read_parquet`), not in Drive. The team plan asks for exactly these files: the other
members match them row by row on TransactionID. They are not in the public GitHub repository, because they
contain the Kaggle labels, which the competition rules do not allow anyone to redistribute.

## How to read a run name

| Part | Means |
|---|---|
| `ft` | FT-Transformer, a model for tables of numbers |
| `sage` | GraphSAGE, a graph model that sees only which card, address, e-mail and device a payment used |
| `A` / `B` | the columns the FT-Transformer saw: A = all 390 columns of the reference paper, B = the same without the 339 "V" columns |
| `shuf` | GraphSAGE trained on randomly shuffled links (a check of whether the real links matter) |
| `seed42`, `seed1`, `seed2` | the same run repeated with a different random start |

For example, `ft_A_seed42` is the FT-Transformer with the V columns, seed 42.

## All 12 runs

Scores on the test payments. The kept epoch is the training round with the best validation ROC-AUC, chosen
before the test set was scored. Minutes are training time on the A100.

| Run | What it is | Test ROC-AUC | Test PR-AUC | Kept epoch | Minutes |
|---|---|---|---|---|---|
| `ft_A_seed42` | FT-Transformer, with V | 0.856 | 0.416 | 1 | 34.1 |
| `ft_A_seed1` | FT-Transformer, with V | 0.854 | 0.396 | 1 | 34.1 |
| `ft_A_seed2` | FT-Transformer, with V | 0.847 | 0.398 | 3 | 40.4 |
| `ft_B_seed42` | FT-Transformer, without V | 0.757 | 0.219 | 1 | 3.2 |
| `ft_B_seed1` | FT-Transformer, without V | 0.751 | 0.202 | 1 | 3.2 |
| `ft_B_seed2` | FT-Transformer, without V | 0.763 | 0.222 | 1 | 3.2 |
| `sage_seed42` | GraphSAGE, real links | 0.754 | 0.141 | 2 | 1.7 |
| `sage_seed1` | GraphSAGE, real links | 0.760 | 0.133 | 1 | 1.7 |
| `sage_seed2` | GraphSAGE, real links | 0.762 | 0.142 | 1 | 1.7 |
| `sage_shuf_seed42` | GraphSAGE, shuffled links | 0.758 | 0.142 | 1 | 2.2 |
| `sage_shuf_seed1` | GraphSAGE, shuffled links | 0.747 | 0.131 | 2 | 2.2 |
| `sage_shuf_seed2` | GraphSAGE, shuffled links | 0.759 | 0.141 | 1 | 2.2 |

- **ROC-AUC:** 0.5 = guessing, 1.0 = perfect.
- **PR-AUC:** guessing gives 0.035, because 3.5% of payments are fraud. This is the main score.

The exact numbers (6 decimals) are in `graph_vs_V/runs.csv`. The graphs (ROC and precision-recall curves,
training curves, overfitting, frauds caught in the top 1% and 5%, and the real-vs-shuffled and A-vs-B
comparisons) are in `01_results.ipynb` in the top folder.
