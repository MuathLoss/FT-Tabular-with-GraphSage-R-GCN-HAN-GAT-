# Fusion results (next step)

Our FT-Transformer combined with each graph model of the team: GraphSAGE (ours), R-GCN, HAN and GAT (from
teammates, in `results/from_teammates/`).

The fusion is done with the team's shared `fusion.py` (top folder), unchanged, so every graph model is fused
the same way:
- score = z(FT-Transformer) + alpha × z(graph model), where z standardises each model's scores with the mean
  and standard deviation of its own validation scores;
- alpha is one of 0, 0.025, ..., 1, chosen on validation PR-AUC (alpha = 0 is the FT-Transformer alone);
- seeds are paired (FT seed 42 with graph seed 42, and so on), for column sets A and B, with the real graph
  and with the shuffled links;
- the test set is scored once, with alpha already fixed, and compared with 2,000 paired bootstrap
  resamples. A graph counts as helping only if the team plan's three checks hold.

Files that will appear here, two per graph model:

| File | Content |
|---|---|
| `fusion_sage_ft.csv` | one row per column set, seed and graph (real or shuffled) |
| `fusion_sage_ft_summary.csv` | one row per column set and metric: means, intervals and the three checks |
| `fusion_rgcn_ft.csv`, `fusion_rgcn_ft_summary.csv` | the same for R-GCN |
| `fusion_han_ft.csv`, `fusion_han_ft_summary.csv` | the same for HAN |
| `fusion_gat_ft.csv`, `fusion_gat_ft_summary.csv` | the same for GAT |
