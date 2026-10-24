"""Recurrent layers built on the custom cells, and the review classifier."""
import torch
import torch.nn as nn

from cells import init_cell, make_cell


def reverse_padded(x, lengths):
    """Reverse each sequence inside its own length, leaving the padding at the end. Self-inverse."""
    B, T = x.shape[:2]
    t = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    L = lengths.to(x.device).unsqueeze(1)
    idx = torch.where(t < L, L - 1 - t, t)
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))


class RecurrentLayer(nn.Module):
    """Unrolls a cell over time. Past the end of a sequence the state is frozen, so the final state of
    every sequence is the state after its last real token (equivalent to packed sequences)."""

    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, x, mask):
        B, T, _ = x.shape
        # unbind (not xp[:, t]) so the backward pass stacks the T gradients once instead of T full-size copies
        xps = self.cell.input_proj(x).unbind(1)
        masks = mask.unsqueeze(-1).unbind(1)
        state = self.cell.init_state(B, x.device)
        outputs = []
        for t in range(T):
            _, new_state = self.cell.step(xps[t], state)
            m = masks[t]
            state = tuple(m * n + (1.0 - m) * s for n, s in zip(new_state, state))
            outputs.append(state[0])
        return torch.stack(outputs, dim=1), state[0]


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, cell="vanilla", emb_dim=128, hidden_size=256, proj_size=None,
                 num_layers=1, bidirectional=False, dropout=0.3, init="default"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bidirectional = bidirectional
        self.layers = nn.ModuleList()
        in_size = emb_dim
        for _ in range(num_layers):
            dirs = nn.ModuleList()
            for _ in range(2 if bidirectional else 1):
                c = make_cell(cell, in_size, hidden_size, proj_size)
                init_cell(c, init)
                dirs.append(RecurrentLayer(c))
            self.layers.append(dirs)
            in_size = c.output_size * len(dirs)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_size, n_classes)

    def forward(self, ids, lengths):
        mask = (ids != 0).float()
        x = self.dropout(self.embedding(ids))
        for k, dirs in enumerate(self.layers):
            out_f, last_f = dirs[0](x, mask)
            if self.bidirectional:
                out_b, last_b = dirs[1](reverse_padded(x, lengths), mask)
                x = torch.cat([out_f, reverse_padded(out_b, lengths)], dim=-1)
                last = torch.cat([last_f, last_b], dim=-1)
            else:
                x, last = out_f, last_f
            if k < len(self.layers) - 1:
                x = self.dropout(x)
        return self.head(self.dropout(last))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    recurrent = sum(p.numel() for p in model.layers.parameters() if p.requires_grad)
    return {"total": total, "recurrent": recurrent, "embedding": model.embedding.weight.numel(),
            "head": sum(p.numel() for p in model.head.parameters())}
