# Fusion results

Our FT-Transformer combined with each graph model of the team: GraphSAGE (ours), R-GCN, HAN and GAT (from
teammates, in `results/from_teammates/`). **Done: R-GCN and HAN. Next: GraphSAGE, then GAT when it arrives.**

Every file here was made by `run_fusion.py`, which runs the team's shared `fusion.py` (top folder)
unchanged, so every graph model is fused the same way:
- score = z(FT-Transformer) + alpha × z(graph model), where z standardises each model's scores with the mean
  and standard deviation of its own validation scores;
- alpha is one of 0, 0.025, ..., 1, chosen on validation PR-AUC (alpha = 0 is the FT-Transformer alone);
- seeds are paired (FT seed 42 with graph seed 42, and so on), for column sets A and B, with the real graph
  and with the shuffled links;
- the test set is scored once, with alpha already fixed, and compared with 2,000 paired bootstrap
  resamples. A graph counts as helping only if the team plan's three checks hold: (1) the 95% interval of
  the gain is above 0, (2) the gain is larger than the spread between the FT-Transformer's seeds, and (3)
  the real links beat the shuffled links.

| File | Content |
|---|---|
| `fusion_<graph>_ft.csv` | written by fusion.py: one row per column set, seed and graph (real or shuffled): alpha, validation PR-AUC, test ROC-AUC and PR-AUC of the graph alone, the FT-Transformer alone and fused, the gains with intervals, and the frauds caught in the top 1% and 5% |
| `fusion_<graph>_ft_summary.csv` | written by fusion.py: one row per column set and metric: seed means and ranges, intervals, the three checks and `graph_helps` |
| `fusion_<graph>_ft_inputs.json` | written by run_fusion.py: SHA-256 of fusion.py, of the 24 score files it read and of both tables, and when it ran |

`02_fusion.ipynb` (top folder) shows all of it as tables and a figure, and runs the fusion again to prove
that the saved tables come out the same.

## Results so far (test set, mean of 3 seeds)

| Graph model | FT-Transformer on | PR-AUC: alone -> fused (gain) | Helps on PR-AUC? | ROC-AUC: alone -> fused (gain) | Helps on ROC-AUC? |
|---|---|---|---|---|---|
| R-GCN | set A (with V) | 0.403 -> 0.415 (+0.0115) | no: gain smaller than the seed spread (0.020); real links not better than shuffled | 0.853 -> 0.865 (+0.0124) | yes |
| R-GCN | set B (no V) | 0.214 -> 0.216 (+0.0018) | no: gain interval includes 0 | 0.757 -> 0.787 (+0.0305) | yes |
| HAN | set A (with V) | 0.403 -> 0.416 (+0.0125) | no: gain smaller than the seed spread (0.020) | 0.853 -> 0.864 (+0.0113) | yes |
| HAN | set B (no V) | 0.214 -> 0.217 (+0.0028) | no: gain smaller than the seed spread; real links not better than shuffled | 0.757 -> 0.788 (+0.0311) | yes |
