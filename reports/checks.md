# Data and graph checks

Written by `python runs.py checks`. Every line is an `assert` in common.py that passed.

## Split file and column sets

- Split file `ieee_sorted_split.parquet`: SHA-256 matches the team's file; 590,540 payments sorted by TransactionDT.

| split | payments | frauds | fraud rate | days |
|---|---|---|---|---|
| train | 413,378 | 14,538 | 3.52% | 1.0-120.8 |
| val | 88,581 | 3,042 | 3.43% | 120.8-152.2 |
| test | 88,581 | 3,083 | 3.48% | 152.2-183.0 |

- Column set A: 390 columns (369 numeric, 21 categorical).
- Column set B: 51 columns (30 numeric, 21 categorical).
- Numeric columns with one value in training (divided by 1, not 0): V107.

## Graph files

- Graph file `graph_files.zip`: SHA-256 matches the team's file; 14,239 nodes, as graph_info.json says.
- Real links: 336,236 (each link once per direction, no self-links, never two entities of the same type); 12,401,340 before repeats removed = 413,378 x 30.
- Payment node files line up with the split file row by row (TransactionID and isFraud).
- Training payments: every node holds one real value; the 'missing' node is used exactly when the value is missing; no training payment uses an 'unknown' node.
- Payments with at least one unknown entity: validation 1.42%, test 1.90%.
- Shuffled link files (seed: links): {42: 563346, 1: 563666, 2: 563380}; the real graph has 336,236.
