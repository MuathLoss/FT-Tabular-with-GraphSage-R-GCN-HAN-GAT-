# Method: FT-Transformer and GraphSAGE

The method paragraphs for the FT-Transformer teammate's 12 runs, as the team plan asks (p. 8): "Write the
method paragraph for FT-Transformer and GraphSAGE, including the GraphSAGE settings he chose." The three
paragraphs below are written to go into the paper as they are. Every number in them comes from the runs'
history files (`results/graph_vs_V/<run>_history.json`), from `reports/checks.md`, or from the code and data
directly (parameter counts, fraud weight). The code that produced the runs is `common.py`, `models.py` and
`runs.py`; the SHA-256 hashes stored in the history files match the files in this folder (checked
2026-11-09).

## FT-Transformer

We use the FT-Transformer (Gorishniy et al., 2021) as the tabular model and train it on two column sets.
Set A holds the 390 columns of the reference paper's Table 2 (TransactionAmt, card1-6, addr1-2,
P_emaildomain, R_emaildomain, DeviceType, DeviceInfo, id_01-id_38 and V1-V339; 369 numeric and 21
categorical). Set B is set A without V1-V339 (51 columns; 30 numeric and 21 categorical). Preprocessing
follows Section 3.2 of the reference paper and uses training payments only. A missing numeric value is filled
with the column's training median, and every numeric column is then standardised with its training mean and
standard deviation. V107 takes a single value in training, so it is divided by 1 instead of 0; it carries no
information but keeps set A at 390 columns. Each categorical column is encoded with the values seen in
training, plus one code for a missing value and one for a value never seen in training. Each numeric column
becomes a 192-dimensional token x_i·w_i + b_i, and each categorical column becomes a lookup in its own
192-dimensional embedding table. A learned [CLS] token is placed first (391 tokens for set A, 52 for set B).
The tokens pass through 3 pre-norm Transformer blocks, each with 8 attention heads and a feed-forward
layer 192 → 384 → 192 with ReLU. Dropout of 0.1 is applied to the attention weights and inside the
feed-forward layer. The fraud score is a linear layer on the [CLS] token after the last block. The model has
1,437,697 parameters on set A and 1,307,521 on set B.

## GraphSAGE

GraphSAGE (Hamilton et al., 2017) runs on the shared entity graph that the SOA member built from the
training payments. The graph has 14,239 nodes, one for each training value of card1, addr1, P_emaildomain,
R_emaildomain, DeviceInfo and DeviceType, plus one "missing" and one "unknown" node per entity type. Two
entities are linked when they appear in the same training payment. Repeated links are removed, which leaves
168,118 entity pairs; each is stored once per direction (336,236 directed links), and message passing uses
each link once. The model sees no payment columns. A node's input is a learned 64-dimensional embedding of
the node plus a learned embedding of its entity type. Two GraphSAGE mean layers follow,
h_v' = W_1 · mean{h_u : u ∈ N(v)} + W_2 · h_v + b, the form used by PyTorch Geometric's `SAGEConv` with mean
aggregation. A node with no links gets a neighbour mean of zero. The first layer is followed by ReLU and
dropout 0.2, and the second has no activation. Both layers run over the whole graph at every training step,
with no neighbour sampling. A payment's score comes from the final states of its 6 entity nodes: they are
concatenated (6 × 64 = 384) and passed through Linear(384, 128) → ReLU → dropout 0.2 → Linear(128, 1). During
training, each (payment, entity) pair is replaced by that entity type's "unknown" node with probability 0.1,
drawn again for every batch. Each training payment is seen once per epoch, so about 10% of the training
entities are swapped each epoch, and the unknown nodes learn a representation. At validation and test time
nothing is swapped: a value not seen in training already points to its type's unknown node in the shared
payment files. This applies to 1.42% of validation payments and 1.90% of test payments (at least one
unknown entity). The model has 977,601 parameters, 911,296 of them in the node embeddings. The team plan
leaves GraphSAGE's settings to us (p. 3 and p. 8). We nevertheless use the mandatory settings for all graph
models (p. 4-5), so that GraphSAGE differs from GAT, R-GCN and HAN only in its layer. Our own choices were
made once, before the final runs, and are the same for every seed: learning rate 0.001, batch size 2,048,
dropout 0.2, gradient clipping at norm 1.0, Xavier-uniform initialisation and full-graph message passing.
No hyperparameter search was run. For the shuffled-graph control, the same model with the same seed is
trained on the SOA member's shuffled link file for that seed (563,346, 563,666 and 563,380 directed links
for seeds 42, 1 and 2). Each payment keeps its real entity nodes; only which entities are linked becomes
random.

## Training and evaluation (both models)

Both models are trained with Adam (learning rate 0.001, batch size 2,048) on binary cross-entropy. Fraud
cases are weighted by √(N0/N1) = √(398,840 / 14,538) = 5.24, computed from the training labels (reference
paper Section 3.4 and Table 3). Gradients are clipped to a global norm of 1.0. Linear and embedding weights
are initialised Xavier-uniform, with zero biases. After every epoch the model scores the validation payments,
and we keep the weights with the best validation ROC-AUC. GraphSAGE trains for 12 epochs. The FT-Transformer
trains for up to 80 epochs and stops once 10 epochs have passed without a new best. The test payments are
scored once, at the end of each run, with the kept weights. Every run is repeated with seeds 42, 1 and 2,
which set the initial weights, the batch order, dropout and the unknown-node swaps. Saved scores are the raw
model outputs before the sigmoid (logits). The runs used PyTorch 2.14.1 (CUDA 13.0) and Python 3.13 on one
NVIDIA A100-SXM4-80GB (Google Colab Pro). They ran in full 32-bit precision (TF32 off) with deterministic GPU
kernels, and two short runs from the same seed gave identical scores. Because of GPU memory, each set A
batch of 2,048 payments was processed as two halves whose gradients are summed before the optimizer step;
mathematically, this is the same update as one full batch. One run took 1.7 minutes for GraphSAGE,
2.2 minutes for GraphSAGE on shuffled links, 3.2 minutes for the FT-Transformer on set B and 34-40 minutes
on set A.

## References to cite

- Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A. (2021). Revisiting deep learning models for
  tabular data. *NeurIPS 2021*.
- Hamilton, W. L., Ying, R. and Leskovec, J. (2017). Inductive representation learning on large graphs.
  *NeurIPS 2017*.
- Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR 2015*.
- Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural
  networks. *AISTATS 2010*.
- The reference paper (Ahmad et al., 2026), for Section 3.2, Section 3.4 and Table 3.

## Notes for Reset (not for the paper as written)

1. **Epochs kept.** Validation ROC-AUC peaked early in every run. The kept epoch was 1-3 for the
   FT-Transformer and 1-2 for GraphSAGE, so the FT-Transformer runs stopped after 11-13 epochs. In
   several runs, validation PR-AUC kept rising after the ROC-AUC peak (for example `sage_seed42` and
   `ft_A_seed2`; `plan.md`, note 4). The team rule selects on ROC-AUC, and we followed it.
2. **The unknown-node swap (decision D5 in `plan.md`)** is our reading of "10% of the training payments'
   entities": each (payment, entity) pair is swapped on its own, missing entities included. GAT, R-GCN and
   HAN should do the same, or the paper should say how each one does it.
3. **The shuffled link files are denser than the real graph** (about 563,000 vs 336,236 directed links), so
   the shuffled control also changes how many neighbours each node has (`plan.md`, note 1).
