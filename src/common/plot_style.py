# Colours and sizes of our figures, used by the notebooks
import matplotlib.pyplot as plt

WIDTH = 14.5 / 2.54
MONTH_COLOR = "#3d3d3a"
PART_COLOR = {"train": "#0072B2", "test": "#D55E00"}
SELECTION_COLOR = {"random": "#009E73", "stratified": "#0072B2", "stratified_balanced": "#D55E00"}
INK, MUTED, GRID, EMPTY = "#1f1f1e", "#6b6b66", "#e4e4e0", "#f0f0ec"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .5, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": .02, "pdf.fonttype": 42,
    "figure.dpi": 150,
})


# Save a figure as pdf and png
def save(fig, folder, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf", dpi=300)
    fig.savefig(folder / f"{name}.png", dpi=300)
