# Score files from teammates

The other graph models of the team study, exactly as their authors sent them (never edited). Our
FT-Transformer is fused with each of them (see `results/fusion/`).

Each model has 12 files: seeds 42, 1 and 2 on the real graph (`<model>_seed<N>`) and on the shuffled links
(`<model>_shuf_seed<N>`), each with a validation file (`_val.parquet`) and a test file (`_test.parquet`).
The format is the same as ours: TransactionID, isFraud and score (the raw output before the sigmoid, a
logit), one row per payment.

| Folder | Graph model | Received | Files |
|---|---|---|---|
| `rgcn/` | R-GCN | 9 November 2026 | 12 score files |
| `han/` | HAN | 9 November 2026 | 12 score files |
| `gat/` | GAT | not yet | - |

Checked on arrival (9 November 2026), for all 24 files:
- every file has 88,581 payments, the same TransactionIDs in the same order and the same isFraud labels as
  the team's split file;
- no missing or infinite scores;
- the scores are logits (they go below 0 and above 1), as the team plan asks.

Not received: the `<run>_history.json` files (epochs, settings and chosen epoch) that the team plan also
asks for. Fusion does not need them.

The `.parquet` files are not in the public GitHub repository: they contain the IEEE-CIS labels, which
Kaggle's competition rules do not allow anyone to redistribute.
