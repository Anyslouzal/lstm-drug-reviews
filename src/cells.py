"""LSTM cell variants implemented from scratch with nn.Linear (no nn.LSTM / nn.LSTMCell).

Every cell splits its pre-activations into two parts:
  * input_proj(x):  W_x* x_t + b_*  for all time steps at once (one big matmul, same maths as per-step)
  * step(xp_t, state): adds the recurrent terms and applies the gate equations for one time step.
The weights of the different gates are stored side by side in a single nn.Linear
(e.g. W_x = [W_xi; W_xf; W_xo; W_xg]); this is the same parameterisation as the equations in the
appendix, just computed with one matrix product instead of four.
"""
import math

import torch
import torch.nn as nn


class VanillaLSTMCell(nn.Module):
    """Eqs. (1)-(6)."""

    n_gates = 4

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.output_size = hidden_size
        self.W_x = nn.Linear(input_size, self.n_gates * hidden_size)  # W_xi, W_xf, W_xo, W_xg (+ biases)
        self.W_h = nn.Linear(hidden_size, self.n_gates * hidden_size, bias=False)  # W_hi, W_hf, W_ho, W_hg

    def input_proj(self, x):
        return self.W_x(x)

    def init_state(self, batch_size, device):
        z = torch.zeros(batch_size, self.hidden_size, device=device)
        return z, z

    def step(self, xp, state):
        h, c = state
        i, f, o, g = (xp + self.W_h(h)).chunk(4, dim=-1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

    def recurrent_weights(self):
        return [self.W_h.weight]


class WMCLSTMCell(VanillaLSTMCell):
    """Working Memory Connections (Landi et al., 2021), Eqs. (7)-(12).

    The gates also read the cell state through a squashed projection tanh(W_c* c):
    input/forget gates see c_{t-1}, the output gate sees the freshly updated c_t.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        self.W_cif = nn.Linear(hidden_size, 2 * hidden_size, bias=False)  # W_ci, W_cf
        self.W_co = nn.Linear(hidden_size, hidden_size, bias=False)

    def step(self, xp, state):
        h, c_prev = state
        zi, zf, zo, zg = (xp + self.W_h(h)).chunk(4, dim=-1)
        ci, cf = torch.tanh(self.W_cif(c_prev)).chunk(2, dim=-1)
        i = torch.sigmoid(zi + ci)
        f = torch.sigmoid(zf + cf)
        g = torch.tanh(zg)
        c = f * c_prev + i * g
        o = torch.sigmoid(zo + torch.tanh(self.W_co(c)))
        h = o * torch.tanh(c)
        return h, (h, c)

    def recurrent_weights(self):
        return [self.W_h.weight, self.W_cif.weight, self.W_co.weight]


class CIFGLSTMCell(VanillaLSTMCell):
    """Coupled Input-Forget Gate (Greff et al., 2017), Eqs. (13)-(18): i_t = 1 - f_t."""

    n_gates = 3

    def step(self, xp, state):
        h, c = state
        f, o, g = (xp + self.W_h(h)).chunk(3, dim=-1)
        f, o, g = torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + (1.0 - f) * g
        h = o * torch.tanh(c)
        return h, (h, c)


class LSTMPCell(VanillaLSTMCell):
    """LSTM with recurrent projection layer (Sak et al., 2014), Eqs. (19)-(25).

    The cell has H memory units but only exposes / feeds back a P-dimensional projection h_t = W_p m_t + b_p.
    """

    def __init__(self, input_size, hidden_size, proj_size):
        nn.Module.__init__(self)
        self.input_size, self.hidden_size, self.proj_size = input_size, hidden_size, proj_size
        self.output_size = proj_size
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(proj_size, 4 * hidden_size, bias=False)  # recurrent weights act on h in R^P
        self.W_p = nn.Linear(hidden_size, proj_size)

    def init_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.proj_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))

    def step(self, xp, state):
        h, c = state
        i, f, o, g = (xp + self.W_h(h)).chunk(4, dim=-1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        m = o * torch.tanh(c)
        h = self.W_p(m)
        return h, (h, c)


CELLS = {"vanilla": VanillaLSTMCell, "wmc": WMCLSTMCell, "cifg": CIFGLSTMCell, "lstmp": LSTMPCell}


def make_cell(name, input_size, hidden_size, proj_size=None):
    if name == "lstmp":
        return LSTMPCell(input_size, hidden_size, proj_size or hidden_size // 2)
    return CELLS[name](input_size, hidden_size)


def init_cell(cell, scheme):
    """Weight initialisation schemes compared in the hyper-parameter study.

    default    : PyTorch nn.Linear default, U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases
    xavier     : Glorot-uniform weights, zero biases
    orthogonal : Glorot-uniform input weights, orthogonal recurrent blocks (one per gate), zero biases
    """
    if scheme == "default":
        return
    recurrent = {id(w) for w in cell.recurrent_weights()}
    for module in cell.modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight
        if scheme == "orthogonal" and id(w) in recurrent:
            block = cell.hidden_size  # one (semi-)orthogonal block per gate
            with torch.no_grad():
                for k in range(math.ceil(w.shape[0] / block)):
                    nn.init.orthogonal_(w[k * block:(k + 1) * block])
        elif scheme in ("xavier", "orthogonal"):
            nn.init.xavier_uniform_(w)
        else:
            raise ValueError(f"unknown init scheme {scheme}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
