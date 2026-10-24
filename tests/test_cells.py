"""Correctness checks for the custom cells. nn.LSTM / nn.LSTMCell are used here ONLY as a reference oracle.

Run: python tests/test_cells.py   (or pytest tests/)
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cells import CIFGLSTMCell, LSTMPCell, VanillaLSTMCell, WMCLSTMCell, init_cell  # noqa: E402
from model import LSTMClassifier, RecurrentLayer, reverse_padded  # noqa: E402

torch.manual_seed(0)
E, H, P, B, T = 7, 5, 3, 4, 6


def reorder_ifog_to_ifgo(w):
    """Our gate order is (i, f, o, g); PyTorch uses (i, f, g, o)."""
    i, f, o, g = w.chunk(4, 0)
    return torch.cat([i, f, g, o], 0)


def test_vanilla_matches_torch_lstmcell():
    cell = VanillaLSTMCell(E, H)
    ref = nn.LSTMCell(E, H)
    with torch.no_grad():
        ref.weight_ih.copy_(reorder_ifog_to_ifgo(cell.W_x.weight))
        ref.weight_hh.copy_(reorder_ifog_to_ifgo(cell.W_h.weight))
        ref.bias_ih.copy_(reorder_ifog_to_ifgo(cell.W_x.bias))
        ref.bias_hh.zero_()
    x = torch.randn(B, T, E)
    state, (h_ref, c_ref) = cell.init_state(B, "cpu"), (torch.zeros(B, H), torch.zeros(B, H))
    xp = cell.input_proj(x)
    for t in range(T):
        _, state = cell.step(xp[:, t], state)
        h_ref, c_ref = ref(x[:, t], (h_ref, c_ref))
    assert torch.allclose(state[0], h_ref, atol=1e-6) and torch.allclose(state[1], c_ref, atol=1e-6)


def test_lstmp_matches_torch_lstm_projection():
    cell = LSTMPCell(E, H, P)
    ref = nn.LSTM(E, H, proj_size=P, batch_first=True)
    with torch.no_grad():
        cell.W_p.bias.zero_()  # nn.LSTM's projection has no bias
        ref.weight_ih_l0.copy_(reorder_ifog_to_ifgo(cell.W_x.weight))
        ref.weight_hh_l0.copy_(reorder_ifog_to_ifgo(cell.W_h.weight))
        ref.bias_ih_l0.copy_(reorder_ifog_to_ifgo(cell.W_x.bias))
        ref.bias_hh_l0.zero_()
        ref.weight_hr_l0.copy_(cell.W_p.weight)
    x = torch.randn(B, T, E)
    out, _ = RecurrentLayer(cell)(x, torch.ones(B, T))
    out_ref, _ = ref(x)
    assert out.shape == (B, T, P)
    assert torch.allclose(out, out_ref, atol=1e-6)


def test_cifg_coupled_gates_and_param_count():
    cell = CIFGLSTMCell(E, H)
    n = sum(p.numel() for p in cell.parameters())
    assert n == 3 * H * (E + H) + 3 * H
    # with f = 1 (huge forget bias) the cell must keep c and ignore the candidate: c_t = c_{t-1}
    with torch.no_grad():
        cell.W_x.bias[:H] = 50.0
    c0 = torch.randn(B, H)
    _, (_, c1) = cell.step(cell.input_proj(torch.randn(B, E)), (torch.zeros(B, H), c0))
    assert torch.allclose(c1, c0, atol=1e-5)


def test_wmc_reduces_to_vanilla_without_cell_connections():
    wmc, van = WMCLSTMCell(E, H), VanillaLSTMCell(E, H)
    van.load_state_dict({k: v for k, v in wmc.state_dict().items() if k.startswith(("W_x", "W_h"))})
    with torch.no_grad():
        wmc.W_cif.weight.zero_()
        wmc.W_co.weight.zero_()
    x = torch.randn(B, T, E)
    a, _ = RecurrentLayer(wmc)(x, torch.ones(B, T))
    b, _ = RecurrentLayer(van)(x, torch.ones(B, T))
    assert torch.allclose(a, b, atol=1e-6)
    assert sum(p.numel() for p in wmc.parameters()) == sum(p.numel() for p in van.parameters()) + 3 * H * H


def test_masking_gives_same_final_state_as_unpadded():
    cell = VanillaLSTMCell(E, H)
    layer = RecurrentLayer(cell)
    x = torch.randn(1, T, E)
    _, last_short = layer(x[:, :3], torch.ones(1, 3))
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float)
    _, last_padded = layer(x, mask)
    assert torch.allclose(last_short, last_padded, atol=1e-6)


def test_reverse_padded_is_self_inverse():
    x = torch.randn(B, T, E)
    lengths = torch.tensor([6, 3, 1, 4])
    r = reverse_padded(x, lengths)
    assert torch.allclose(r[1, :3], x[1, :3].flip(0)) and torch.allclose(r[1, 3:], x[1, 3:])
    assert torch.allclose(reverse_padded(r, lengths), x)


def test_init_schemes_and_classifier_forward():
    for scheme in ("default", "xavier", "orthogonal"):
        for cell in ("vanilla", "wmc", "cifg", "lstmp"):
            m = LSTMClassifier(50, 3, cell=cell, emb_dim=E, hidden_size=8, proj_size=4, num_layers=2,
                               bidirectional=True, init=scheme)
            ids = torch.randint(1, 50, (B, T))
            ids[1, 4:] = 0
            out = m(ids, (ids != 0).sum(1))
            assert out.shape == (B, 3) and torch.isfinite(out).all()
    c = VanillaLSTMCell(E, 8)
    init_cell(c, "orthogonal")
    blk = c.W_h.weight[:8]
    assert torch.allclose(blk @ blk.T, torch.eye(8), atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
