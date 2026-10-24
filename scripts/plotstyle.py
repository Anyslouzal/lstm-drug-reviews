"""Shared matplotlib style for all figures of the report and slides."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
# one fixed colour per architecture (categorical slots, never re-ordered)
CELL_COLORS = {"vanilla": "#2a78d6", "wmc": "#eb6834", "cifg": "#1baf7a", "lstmp": "#eda100"}
CELL_NAMES = {"vanilla": "Vanilla LSTM", "wmc": "WMC-LSTM", "cifg": "CIFG-LSTM", "lstmp": "LSTMP"}
# sentiment is a polarity: red pole / gray midpoint / blue pole
CLASS_COLORS = ["#e34948", "#a8a7a1", "#2a78d6"]
CLASS_NAMES = ["negative", "neutral", "positive"]


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
