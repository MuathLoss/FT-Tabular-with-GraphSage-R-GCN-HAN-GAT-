# Data (not included)

This project uses the **IEEE-CIS Fraud Detection** data from Kaggle:
<https://www.kaggle.com/competitions/ieee-fraud-detection/data>. To get it, sign in to Kaggle and accept the
competition rules. Those rules allow the data to be used "for non-commercial purposes only, including ...
for academic research and education", and do not allow anyone to publish or redistribute it. So no file
that contains the data is in this repository, including:

| File (in `data/processed/`) | What it is | Made by |
|---|---|---|
| `ieee_sorted_split.parquet` | Kaggle's `train_transaction.csv` with the columns of `train_identity.csv` added by TransactionID (left join): 590,540 payments sorted by TransactionDT, with a `split` column: the first 70% `train` (413,378), the next 15% `val` (88,581) and the last 15% `test` (88,581) | the team (shared file) |
| `graph_files.zip` | the shared entity graph: 14,239 nodes (one per training value of card1, addr1, P_emaildomain, R_emaildomain, DeviceInfo and DeviceType, plus one "missing" and one "unknown" node per type), the links between entities that share a training payment, each payment's 6 node numbers, and 3 shuffled-link versions (seeds 42, 1 and 2) | a teammate (shared file) |
| `results/**/*.parquet` | every run's score for every validation and test payment (TransactionID, isFraud, score) | the runs |

The code checks the SHA-256 hash of both shared files before it does anything (`common.py`), so a run only
starts with exactly the files the team used:

| File | SHA-256 |
|---|---|
| `ieee_sorted_split.parquet` | `9cde42fd5c67ed7edb7f76ab7f8b3d7d7e2c4e789cb960efc5c466182875dbe1` |
| `graph_files.zip` | `5d0d5f6534da40fc1d0da3205ee158c54f2320ef2c2c3814a8b70ae63031182b` |

`reports/checks.md` lists what was checked in both files (row counts, time order, column sets, graph layout).
