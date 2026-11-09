# Plan and decisions: the FT-Transformer teammate's part of the team study

The team plan is `references/Team_plan_graph_vs_V.pdf` (21 Oct 2026). This file lists what this folder
runs, every setting, and every choice the plan leaves to us. The team's fixed settings change only if the
whole team agrees.

## What this folder does

**12 training runs** and their hand-in files. Reset does the fusion and evaluation; nothing here is fused.

| Runs | Model | Inputs |
|---|---|---|
| `ft_A_seed42`, `ft_A_seed1`, `ft_A_seed2` | FT-Transformer | Column set A: paper Table 2, 390 columns, with V1-V339 |
| `ft_B_seed42`, `ft_B_seed1`, `ft_B_seed2` | FT-Transformer | Column set B: A without V1-V339, 51 columns |
| `sage_seed42`, `sage_seed1`, `sage_seed2` | GraphSAGE | The SOA member's entity graph; no payment columns |
| `sage_shuf_seed42`, `sage_shuf_seed1`, `sage_shuf_seed2` | GraphSAGE | The SOA member's shuffled links for the same seed |

Every run hands in three files, plus one row in `results/graph_vs_V/runs.csv`:
- `<run>_val.parquet` and `<run>_test.parquet`: TransactionID, isFraud and score, the raw output before
  the sigmoid (a logit);
- `<run>_history.json`: every epoch's loss and validation ROC-AUC and PR-AUC, plus the settings and the
  chosen epoch.

## Settings

| Setting | FT-Transformer | GraphSAGE | Source |
|---|---|---|---|
| Data | `ieee_sorted_split.parquet` | same | Team plan: nobody makes their own split |
| Split | 413,378 / 88,581 / 88,581 by time | same | Paper 3.2 |
| Inputs | Set A or set B | 6 entity nodes per payment | Paper Table 2 / team plan |
| Preprocessing | Training medians, mean and std; categories from training; unseen values become "unknown" | none | Paper 3.2 |
| Graph | none | `graph_files.zip`: 14,239 nodes, 336,236 links (each once per direction) | Team plan: nobody builds their own |
| Architecture | 192-number tokens, 3 pre-norm blocks, 8 heads, feed-forward 384, [CLS] head | Node embedding + entity-type embedding (64); 2 GraphSAGE mean layers; head 384 -> 128 -> 1 | Ours (FT) / team plan (GraphSAGE) |
| Unknown nodes | none | 10% of training entities swapped for their type's unknown node | Team plan |
| Epochs | Up to 80; stop 10 epochs after the best | 12 | Paper Table 3 / team plan |
| Chosen epoch | Best validation ROC-AUC | Best validation ROC-AUC | Team plan |
| Optimizer, loss | Adam; BCE with fraud weight sqrt(N0/N1) from training | same | Paper Table 3, 3.4 |
| Seeds | 42, 1, 2 | 42, 1, 2 | Team plan |
| Test set | Scored once, at the end of each run | same | Team plan |
| Free choices (same for every seed) | lr 0.001, batch 2,048, dropout 0.1, gradient clipping 1.0, Xavier weights | lr 0.001, batch 2,048, dropout 0.2, gradient clipping 1.0, Xavier weights | Ours |
| Precision | Full 32-bit (TF32 off), deterministic GPU kernels | same | Ours |

## Check against the team plan

Every rule in the team plan that applies to the FT-Transformer teammate, and where the code follows it.
Checked on 2026-11-09.

| Team plan rule (page) | Where in the code | Status |
|---|---|---|
| Use Reset's split file; nobody makes their own (p. 2) | `common.load_split()` refuses any other file (SHA-256) | ✓ |
| 590,540 payments, 70/15/15 by TransactionDT: 413,378 / 88,581 / 88,581 (p. 2) | `load_split()` asserts the sizes, the time order and where each split sits | ✓ (`checks.md`) |
| Column set A = Table 2, 390 columns; B = A without V1-V339, 51 (p. 2) | `common.column_set()` asserts 390 and 51 | ✓ |
| Preprocessing as paper 3.2: training medians, mean and std, categories from training, unseen -> unknown (p. 3, 8) | `numeric_matrix()`, `category_codes()`; compared with the paper's Section 3.2 text | ✓ (D3, D4) |
| Adam; weighted BCE, fraud weight sqrt(N0/N1) from training (p. 3, 5) | `models.training_setup()` | ✓ |
| Early stopping on validation ROC-AUC, patience 10; tabular up to 80 epochs (p. 3, 8) | `models.fit()` keeps the best validation ROC-AUC; `FT_EPOCHS = 80`, `FT_PATIENCE = 10` | ✓ |
| Test scored once, at the end of each run (p. 3, 5) | `runs.train()` scores the test set only after the best epoch is restored | ✓ |
| Seeds 42, 1, 2 for every run (p. 3, 5) | `SEEDS = [42, 1, 2]` | ✓ |
| Saved score = the logit before the sigmoid (p. 3, 5) | `models.predict()` returns raw outputs; `save_run()` saves them | ✓ |
| GraphSAGE on the SOA member's graph file; nobody builds their own (p. 3, 4) | `common.load_graph()` reads `graph_files.zip` unchanged (SHA-256) | ✓ |
| 14,239 nodes, 12,401,340 links before repeats removed (p. 4, 8) | asserted in `load_graph()` and `graph_checks()` | ✓ |
| Message passing uses each link once (p. 4) | `GraphSAGE.neighbour_mean()` over the unique links in `graph_edges.npy` | ✓ |
| Unseen values -> their type's unknown node; missing -> the missing node (p. 4) | built into the SOA payment files; checked in `graph_checks()` | ✓ |
| 10% of training entities swapped for unknown, each epoch (p. 4) | `GraphSAGE.forward()` in training mode | ✓ (our reading: D5) |
| Node inputs = node embedding + entity-type embedding; no payment columns (p. 4) | `GraphSAGE.node_states()`; the model gets only node numbers | ✓ |
| Payment score: 6 x 64 = 384 -> Linear(384, 128) -> ReLU -> dropout -> Linear(128, 1) (p. 4) | `GraphSAGE.head` | ✓ |
| 2 layers, embedding size 64, 12 epochs, keep the best validation ROC-AUC (p. 5) | `SAGELayer` x 2, `dim=64`, `SAGE_EPOCHS = 12` | ✓ |
| Shuffled control: the same model on the SOA member's shuffled file for the same seed (p. 5, 11) | `sage_shuf_seed<N>` loads `graph_shuf_seed<N>_edges.npy` with seed N | ✓ |
| Free choices chosen once, the same for all seeds, written in the history file (p. 5) | constants in `runs.py`; every setting goes into `<run>_history.json` | ✓ |
| FT-Transformer on A and B, 3 seeds each: 6 runs (p. 8) | `ft_A_seed<N>`, `ft_B_seed<N>` | ✓ (done 2026-11-09, `runs.csv`) |
| GraphSAGE 3 seeds + 3 seeds on the shuffled files: 6 runs (p. 8) | `sage_seed<N>`, `sage_shuf_seed<N>` | ✓ (done 2026-11-09, `runs.csv`) |
| Hand-in: `<run>_val.parquet`, `<run>_test.parquet` (TransactionID, isFraud, score) and `<run>_history.json` (p. 13-14) | `common.save_run()`; format checked on a scratch test run | ✓ |
| One row per run: run name, test ROC-AUC, test PR-AUC, best epoch, minutes (p. 14) | `results/graph_vs_V/runs.csv` (plus validation scores and finish time) | ✓ |
| Run names `<model>_<set>_seed<N>`, `<model>_seed<N>`, `<model>_shuf_seed<N>` (p. 14) | `ft_A_seed42`, `sage_seed42`, `sage_shuf_seed42` | ✓ |
| Method paragraph for the FT-Transformer and GraphSAGE, with the chosen settings (p. 8) | `reports/method.md` | ✓ |
| Fusion, bootstrap comparisons, the paper's hybrid, the other members' models | Reset's and the other members' tasks; not in this folder | not ours |

Graphs of the 12 runs: `01_results.ipynb` (code in `report.py`), made from their saved validation and test
results; the figures are saved to `results/figures/`. They are for us; they do not replace Reset's fusion and
evaluation.

## Decisions

| # | Date | Decision |
|---|---|---|
| D1 | 2026-11-09 | **We switched to this team plan**, at the user's request, replacing the earlier solo study: GraphSAGE with payments inside the graph, with and without V, plus fusion heads. That study's record is kept in `D:\Desktop\claudepaper_old_V_study.zip` (code, reports, tables, test metrics); its other files were deleted. |
| D2 | 2026-11-09 | **Data:** Reset's split file, hash-checked on every load. It has the same payments, order, labels and split as our earlier split. Its numbers are stored as float32, so they differ from the raw CSVs in the 8th significant digit at most (worst relative difference 6e-8). |
| D3 | 2026-11-09 | **Set A keeps V107**, giving the plan's 390 columns. V107 has one value in training, so it is divided by 1 instead of 0 and carries no information. |
| D4 | 2026-11-09 | **Categorical codes:** 0 = missing, 1 = a value not seen in training ("unknown"), 2+ = training values (paper 3.2). |
| D5 | 2026-11-09 | **"10% of entities swapped for unknown":** each (payment, entity) pair independently, with probability 0.1, drawn again every batch. Each payment is seen once per epoch, so this is 10% per epoch. Missing entities are included. **To confirm with the SOA member**, so GAT, R-GCN and HAN do the same. |
| D6 | 2026-11-09 | **GraphSAGE layers:** ReLU and dropout between the two layers, nothing after the second. A node with no links gets a neighbour mean of zero. |
| D7 | 2026-11-09 | **FT-Transformer:** the same architecture as in our earlier study, since the plan leaves it free. Each 2,048-payment batch is split into smaller GPU steps only to fit in memory; the maths is the same. Set B always uses 1 step. Set A uses 8 steps on an 8 GB GPU, 4 on 20 GB or more, and 2 on 39 GB or more. |
| D8 | 2026-11-09 | **Training runs on Colab Pro (A100 or L4) from the Google Drive folder.** All 3 seeds of a model run on the same GPU type. An interrupted run continues only with the same code, data, settings and GPU type; otherwise it must restart. |
| D9 | 2026-11-09 | **Run names:** the GraphSAGE runs are `sage_seed<N>` and `sage_shuf_seed<N>`, following style guide section 10 and the plan's `<model>_seed<N>` pattern. |
| D10 | 2026-11-09 | **Folder tidy after training:** `colab.ipynb` renamed `00_train_runs.ipynb`; `01_results.ipynb` no longer writes a `__pycache__` folder; plain-language guides in `README.md` and `results/README.md`. The run files stay in `results/graph_vs_V/` under their team names, because the code, the results notebook and the team's fusion find them there (style guide section 2: copy, don't move). |
| D11 | 2026-11-09 | **Public GitHub repository** (`MuathLoss/FT-Tabular-with-GraphSage-R-GCN-HAN-GAT-`, MIT licence for the code). Kaggle's IEEE-CIS rules allow the data for non-commercial use only (including academic research and education) and forbid redistributing it, so the split file, the graph files and every score file (TransactionID + isFraud) stay out; trained models, histories, logs, figures and reports are published for non-commercial research and education. The team plan PDF, the code style guide and the reference paper PDF are not published (cited in `references/README.md`). The team's `fusion.py` is included unchanged, with the team's agreement. |
| D12 | 2026-11-09 | **Fusion runs through `run_fusion.py`**, because `fusion.py` (which must stay unchanged) reads every score file from one folder and writes its tables there, while our files are split between `results/graph_vs_V/` and `results/from_teammates/`. It copies the 24 needed files into a temporary folder, runs `fusion.py`, checks that its FT-Transformer-alone test scores equal `runs.csv`, and saves both tables plus the SHA-256 of every input to `results/fusion/` (write-once). Run on the PC (CPU, about 3 minutes per graph model); `02_fusion.ipynb` shows the results and re-runs the fusion to prove they come out the same. |

## Notes for the team

These came out of reviewing the plan; none of them changes a team setting.

1. **The shuffled link files have about 67% more links than the real graph** (about 563k vs 336k).
   Shuffling each column on its own creates many new pairs, so the shuffled control is also a denser graph.
2. **The shuffled control keeps each payment's real nodes**, and every graph model scores a payment from
   its own nodes' learned vectors. So the control removes only what neighbours contribute; the model can still
   remember individual cards and devices. If real ≈ shuffled, message passing adds nothing beyond that.
3. **The "12,401,340 links" check is weak:** it is just 413,378 × 30, which any graph linking 6 entities
   per payment gives. The node count and the file checks in `reports/checks.md` are stronger.
4. **Choosing the epoch by ROC-AUC picks earlier epochs than PR-AUC.** In our earlier FT-Transformer run on
   set A, ROC-AUC would have picked epoch 2 (validation PR-AUC 0.420) instead of epoch 6 (0.445). It is the
   same for GraphSAGE (`sage_seed42`, 2026-11-09): validation ROC-AUC peaks at epoch 2 (0.789) and then
   falls, while validation PR-AUC keeps rising to 0.227 at epoch 12; the team rule keeps epoch 2 (PR-AUC
   0.190). We follow the rule. This is validation data only; no test score was used to notice it.

## Order of work

1. `python runs.py checks`: the split file, column sets and graph files (see `reports/checks.md`). Done.
2. `python runs.py smoke` on this PC: run times, and two identical runs give identical scores. Done.
3. In Colab, on the A100 or L4: `checks` and `smoke` again, then `all`, but only after the user's go-ahead.
   Done on 2026-11-09 on an A100: all 12 runs finished between 13:39 and 15:50 UTC.
4. Write the method paragraph for the FT-Transformer and GraphSAGE (`reports/method.md`). Done.
5. Run `01_results.ipynb` for the graphs, then hand in to Reset: the 24 score files, the 12 history files,
   `runs.csv` and `reports/method.md`.
6. Fusion: the FT-Transformer with GraphSAGE, R-GCN, HAN and GAT, using the team's `fusion.py` unchanged
   (`python run_fusion.py <graph>`); results in `results/fusion/`, shown and re-checked in `02_fusion.ipynb`.
   R-GCN, HAN and GraphSAGE done on 2026-11-09 (teammates' scores received and checked in
   `results/from_teammates/`). GAT when its scores arrive.
