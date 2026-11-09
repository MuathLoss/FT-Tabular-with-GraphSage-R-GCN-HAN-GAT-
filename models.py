"""The FT-Transformer teammate's two models (team plan p. 8), with one fit() and one predict() for both.
No experiments here; runs.py calls these with different inputs.

FTTransformer (Gorishniy et al. 2021), the tabular model, on column set A or B:
  each numeric column -> a token x * w + b; each categorical column -> its own lookup table  -> FTTransformer
  a learned [CLS] token first; 3 pre-norm blocks, 8 heads, feed-forward 192 -> 384 -> 192    -> TransformerBlock
  score = Linear(192 -> 1) on the [CLS] token after the blocks
  (the same architecture as in our earlier study; the team plan leaves its settings free)
GraphSAGE on the SOA member's entity graph, following "Mandatory settings for all graph models" (team plan p. 4-5):
  node input = learned embedding of each node + embedding of its entity type (64 numbers)   -> GraphSAGE.node_states()
  2 GraphSAGE mean layers, each link used once (Hamilton et al. 2017; paper 3.3.3)          -> SAGELayer, neighbour_mean()
  payment score = its 6 entity nodes joined (6 x 64 = 384) -> Linear(384, 128) -> ReLU -> dropout -> Linear(128, 1)
  each epoch, 10% of the training payments' entities are swapped for their type's "unknown" node,
  so that node learns something: each (payment, entity) pair independently, drawn again every batch -> GraphSAGE.forward()
  no payment columns at all; validation and test payments only look up their nodes
Both, from the paper (Table 3, 3.4): Adam; weighted BCE with fraud weight sqrt(N0/N1) from training labels; keep
the epoch with the best validation ROC-AUC (team rule; the FT-Transformer stops 10 epochs after its best).
Our choices where the plan leaves them free, the same for every seed and written in each history file: learning
rate 0.001, batch 2,048, dropout 0.1 (FT) and 0.2 (GraphSAGE), gradient clipping at 1.0, Xavier-uniform weights,
full 32-bit precision (TF32 off), deterministic GPU kernels. Every saved score is the raw output before the
sigmoid (a logit).
"""
import copy
import io
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic GPU matrix products
import numpy as np
import torch
from torch import nn

from common import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """One seed for Python, NumPy and PyTorch (CPU and GPU), deterministic GPU kernels, and full 32-bit
    matrix products (an A100 would otherwise use the faster, less precise TF32 format)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def init_weights(model):
    """Xavier-uniform weights and zero biases for every layer; call after set_seed()."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


class SAGELayer(nn.Module):
    """GraphSAGE mean layer (Hamilton et al. 2017): h_v' = W_l mean(h_u : u in N(v)) + W_r h_v. Same formula
    as PyG's SAGEConv(aggr="mean"); the caller computes the neighbours' mean."""

    def __init__(self, dim):
        super().__init__()
        self.lin_l = nn.Linear(dim, dim)              # neighbours' mean, with the layer's bias
        self.lin_r = nn.Linear(dim, dim, bias=False)  # the node itself

    def forward(self, h, neighbour_mean):
        return self.lin_l(neighbour_mean) + self.lin_r(h)


class GraphSAGE(nn.Module):
    """GraphSAGE on the shared entity graph (team plan p. 4-5). `edges` [2, E]: each link once per direction;
    `node_type`: each node's entity type; `unknown`: the "unknown" node of each of the 6 entity columns.
    Input: each payment's 6 entity node numbers. Output: one logit per payment."""

    def __init__(self, edges, node_type, unknown, dim=64, dropout=0.2, swap=0.1):
        super().__init__()
        degree = torch.bincount(edges[1], minlength=len(node_type)).clamp(min=1).float()[:, None]
        for name, tensor in [("source", edges[0]), ("target", edges[1]), ("degree", degree),
                             ("node_type", node_type), ("unknown", unknown)]:
            self.register_buffer(name, tensor, persistent=False)   # the fixed graph, not part of the checkpoint
        self.node_embedding = nn.Embedding(len(node_type), dim)
        self.type_embedding = nn.Embedding(int(node_type.max()) + 1, dim)
        self.layers = nn.ModuleList([SAGELayer(dim), SAGELayer(dim)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(6 * dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.swap = swap
        init_weights(self)

    def neighbour_mean(self, h):
        """Each node's mean over its linked nodes, each link once; a node with no link gets zeros. Indexed
        sums (index_add) are reproducible on the GPU with deterministic kernels on."""
        return torch.zeros_like(h).index_add(0, self.target, h[self.source]) / self.degree

    def node_states(self):
        """Every node's 64 numbers after the two GraphSAGE layers (ReLU and dropout between them)."""
        h = self.node_embedding.weight + self.type_embedding(self.node_type)
        h = self.dropout(torch.relu(self.layers[0](h, self.neighbour_mean(h))))
        return self.layers[1](h, self.neighbour_mean(h))

    def forward(self, node_ids):
        if self.training:   # team rule: 10% of training entities become their type's unknown node
            swapped = torch.rand(node_ids.shape, device=node_ids.device) < self.swap
            node_ids = torch.where(swapped, self.unknown.expand_as(node_ids), node_ids)
        return self.head(self.node_states()[node_ids].flatten(1)).squeeze(1)


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attention(LayerNorm(x)), then x + FeedForward(LayerNorm(x)). Dropout on the
    attention weights and inside the feed-forward, none on the residual path."""

    def __init__(self, dim, heads, hidden, dropout):
        super().__init__()
        self.norm_attention, self.norm_feed_forward = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x):
        u = self.norm_attention(x)
        x = x + self.attention(u, u, u, need_weights=False)[0]
        return x + self.feed_forward(self.norm_feed_forward(x))


class FTTransformer(nn.Module):
    """FT-Transformer (Gorishniy et al. 2021). Each numeric column becomes a token x_i * w_i + b_i and each
    categorical column a lookup in its own table, all with `dim` numbers; a learned [CLS] token goes first.
    After the blocks, Linear(dim -> 1) on the [CLS] token gives the fraud score."""

    def __init__(self, n_numeric, n_categories, dim=192, blocks=3, heads=8, hidden=384, dropout=0.1):
        super().__init__()
        self.numeric_weight = nn.Parameter(torch.empty(n_numeric, dim))
        self.numeric_bias = nn.Parameter(torch.zeros(n_numeric, dim))
        self.embed = nn.ModuleList(nn.Embedding(n, dim) for n in n_categories)
        self.cls = nn.Parameter(torch.empty(1, dim))
        self.blocks = nn.ModuleList(TransformerBlock(dim, heads, hidden, dropout) for _ in range(blocks))
        self.head = nn.Linear(dim, 1)
        init_weights(self)
        nn.init.xavier_uniform_(self.numeric_weight)
        nn.init.xavier_uniform_(self.cls)

    def forward(self, x, codes):
        tokens = torch.cat([self.cls.expand(len(x), 1, -1), x[..., None] * self.numeric_weight + self.numeric_bias,
                            torch.stack([emb(codes[:, k]) for k, emb in enumerate(self.embed)], 1)], 1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0]).squeeze(1)


def train_epoch(model, inputs, y_train, loss_fn, optimizer, order, batch_size, pieces):
    """One pass over the training payments in a seeded random order. pieces > 1 splits each batch into that
    many smaller GPU steps whose gradients add up to the same batch (GPU memory only, same maths).
    Returns the mean loss and every payment's training-mode score (dropout on, weights still changing)."""
    model.train()
    total, scores = 0.0, torch.empty(len(y_train), device=DEVICE)
    for batch in torch.randperm(len(y_train), generator=order).split(batch_size):
        optimizer.zero_grad()
        for part in batch.chunk(pieces):
            logits = model(*(t[part].to(DEVICE) for t in inputs))
            loss = loss_fn(logits, y_train[part]) * (len(part) / len(batch))
            loss.backward()
            total += loss.item() * len(batch)
            scores[part.to(DEVICE)] = logits.detach()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return total / len(y_train), scores.cpu().numpy()


def training_setup(model, y_train, lr, seed):
    """Weighted BCE with fraud weight sqrt(N0/N1) from the training labels (paper 3.4), Adam (Table 3), and
    the batch order, seeded so that every run with this seed sees the same order."""
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.sqrt((y_train == 0).sum() / y_train.sum()))
    return loss_fn, torch.optim.Adam(model.parameters(), lr=lr), torch.Generator().manual_seed(seed)


def save_progress(path, fingerprint, history, best, model, optimizer, order):
    """Everything needed to continue exactly after an interruption (a Colab disconnect): weights, optimizer,
    random states, history and the best weights so far. Written to a temporary file and then swapped in, so
    a crash while saving never leaves a broken file. A working file, deleted when the run is saved."""
    state = {"fingerprint": fingerprint, "history": history, "best": best, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "order": order.get_state(), "cpu_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def resume_progress(path, fingerprint, model, optimizer, order):
    """Continue from a progress file. The code, data and settings must be the same (fingerprint); then
    weights, optimizer and random states are restored. Returns the history and the best weights so far."""
    state = torch.load(path, map_location=DEVICE)
    assert state["fingerprint"] == fingerprint, "code, data or settings changed since the interrupted run"
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    order.set_state(state["order"].cpu())   # random states must come back as CPU tensors
    torch.set_rng_state(state["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    print(f"  resumed after epoch {len(state['history'])} from {path.name}", flush=True)
    return state["history"], tuple(state["best"])


def fit(model, data, y, epochs, seed, batch_size, lr, progress, fingerprint, patience=None, pieces=1, eval_batch=16384):
    """Train with Adam on weighted BCE (training_setup). After each epoch, score validation and keep the
    weights with the best validation ROC-AUC (team rule; a tie keeps the earlier epoch); with `patience`, stop
    after that many epochs without a new best. Every epoch records loss, training and validation ROC-AUC and
    PR-AUC, and saves the full state to `progress`, so an interrupted run continues exactly. `data[split]` is a
    tuple of tensors with one row per payment; only "train" and "val" are used here. Returns history, best epoch."""
    y_train = torch.as_tensor(y["train"], dtype=torch.float32, device=DEVICE)
    loss_fn, optimizer, order = training_setup(model, y_train, lr, seed)
    history, best = [], (-1.0, 0, None)
    if progress.exists():
        history, best = resume_progress(progress, fingerprint, model, optimizer, order)
    for epoch in range(len(history) + 1, epochs + 1):
        if patience and epoch - 1 - best[1] >= patience:
            break
        start = time.time()
        loss, train_scores = train_epoch(model, data["train"], y_train, loss_fn, optimizer, order, batch_size, pieces)
        train, val = evaluate(y["train"], train_scores), evaluate(y["val"], predict(model, data["val"], eval_batch))
        history.append({"epoch": epoch, "loss": loss, "train_roc_auc": train["roc_auc"], "train_pr_auc": train["pr_auc"],
                        "val_roc_auc": val["roc_auc"], "val_pr_auc": val["pr_auc"], "seconds": round(time.time() - start, 1)})
        print(f"  epoch {epoch:2d} | loss {loss:.4f} | val ROC-AUC {val['roc_auc']:.4f} | val PR-AUC {val['pr_auc']:.4f}"
              f" | {time.time() - start:.0f} s", flush=True)
        if val["roc_auc"] > best[0]:
            best = (val["roc_auc"], epoch, copy.deepcopy(model.state_dict()))
        save_progress(progress, fingerprint, history, best, model, optimizer, order)
    model.load_state_dict(best[2])
    return history, best[1]


@torch.no_grad()
def predict(model, inputs, batch_size=16384):
    """Raw scores (logits, before the sigmoid) in eval mode, so dropout is off and no entity is swapped.
    Inputs sit in RAM; each batch is copied to the GPU and its scores come straight back."""
    model.eval()
    return torch.cat([model(*(t[i:i + batch_size].to(DEVICE) for t in inputs)).cpu()
                      for i in range(0, len(inputs[0]), batch_size)]).numpy()


def checkpoint_bytes(model):
    """The model's weights as bytes, for common.save_run()."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()
