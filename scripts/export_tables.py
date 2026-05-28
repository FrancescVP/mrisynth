"""Export experiment result tables as PNG images."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("figures/tables")
OUT.mkdir(parents=True, exist_ok=True)

BEST_COLOR  = "#d4edda"   # soft green
WORST_COLOR = "#f8d7da"   # soft red
HEAD_COLOR  = "#2c3e50"   # dark slate
ALT_COLOR   = "#f7f9fc"   # light row stripe
WHITE       = "#ffffff"
TEXT_HEAD   = "white"
TEXT_BODY   = "#1a1a2e"


def render_table(title, subtitle, columns, rows, highlight_rows=None,
                 highlight_color=None, worst_rows=None, filename="table.png",
                 col_widths=None):
    """Render a table with title and save as PNG."""
    n_rows = len(rows)
    n_cols = len(columns)

    fig_w = sum(col_widths) if col_widths else n_cols * 2.2
    fig_h = 0.5 + n_rows * 0.42 + 0.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Table fills the whole figure
    top = 0.97
    row_h = (top - 0.04) / (n_rows + 1)
    left = 0.02
    total_w = 0.96

    # Column widths as fractions
    if col_widths:
        total = sum(col_widths)
        fracs = [w / total * total_w for w in col_widths]
    else:
        fracs = [total_w / n_cols] * n_cols

    x_starts = [left]
    for f in fracs[:-1]:
        x_starts.append(x_starts[-1] + f)

    def draw_cell(fig_, row_idx, col_idx, text, bg, fg=TEXT_BODY,
                  bold=False, fontsize=9, align="center"):
        x = x_starts[col_idx]
        w = fracs[col_idx]
        y = top - (row_idx + 1) * row_h
        rect = mpatches.FancyBboxPatch(
            (x, y), w, row_h,
            boxstyle="square,pad=0",
            linewidth=0.4, edgecolor="#cccccc",
            facecolor=bg,
            transform=fig_.transFigure, figure=fig_
        )
        fig_.add_artist(rect)
        ha = "left" if align == "left" else "center"
        tx = x + 0.012 if align == "left" else x + w / 2
        fig_.text(tx, y + row_h / 2, text, ha=ha, va="center",
                  fontsize=fontsize, color=fg,
                  fontweight="bold" if bold else "normal",
                  transform=fig_.transFigure)

    # Header
    for c, col in enumerate(columns):
        draw_cell(fig, -1, c, col, HEAD_COLOR, fg=TEXT_HEAD,
                  bold=True, fontsize=9.5)

    # Rows
    highlight_set = set(highlight_rows or [])
    worst_set     = set(worst_rows or [])

    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            if r in highlight_set:
                bg = highlight_color or BEST_COLOR
            elif r in worst_set:
                bg = WORST_COLOR
            elif r % 2 == 1:
                bg = ALT_COLOR
            else:
                bg = WHITE
            align = "left" if c == 0 else "center"
            bold  = r in highlight_set and c == 0
            draw_cell(fig, r, c, val, bg, bold=bold, fontsize=8.5,
                      align=align)

    plt.savefig(OUT / filename, dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Saved {OUT / filename}")


# ── Table 1: Architecture sweep ──────────────────────────────────────────────
render_table(
    title="Sweep 1 — Architecture",
    subtitle="Fixed: L1 loss · lr = 1e-4 · 300 epochs | Best SSIM highlighted in green",
    columns=["Experiment", "Params", "Channels", "Best SSIM", "Best MAE"],
    rows=[
        ["rflow_tiny",   "19.2 M", "[32, 64, 128, 128]",     "0.740", "0.0699"],
        ["rflow_small",  "44.7 M", "[32, 64, 128, 256]",     "0.741", "0.0695"],
        ["rflow_medium", "76.5 M", "[64, 128, 256, 256]",    "0.762", "0.0698"],
        ["rflow_deep",   "76.4 M", "[32, 64, 128, 256, 256]","0.745", "0.0732"],
        ["rflow_large",  "178.6 M","[64, 128, 256, 512]",    "0.767", "0.0650"],
    ],
    highlight_rows=[4],
    col_widths=[2.8, 1.8, 3.8, 1.8, 1.8],
    filename="table1_architecture.png",
)

# ── Table 2: HP sweep ─────────────────────────────────────────────────────────
render_table(
    title="Sweep 2 — Hyperparameters",
    subtitle="Fixed: medium arch [64,128,256,256] · L1 loss · 300 epochs | Best highlighted in green",
    columns=["Experiment", "Change vs Baseline", "Best SSIM", "Best MAE"],
    rows=[
        ["hp_lr_high",  "lr = 2e-4  ← BEST SSIM",          "0.769", "0.0661"],
        ["hp_resblk3",  "3 res blocks / level",             "0.767", "0.0688"],
        ["hp_attn3",    "3 attention levels",               "0.765", "0.0637"],
        ["hp_baseline", "lr = 1e-4  (reference)",           "0.764", "0.0651"],
        ["hp_decay",    "linear LR decay 200 + 100 ep",     "0.762", "0.0642"],
        ["hp_ts250",    "250 training timesteps  ← BEST MAE","0.762", "0.0629"],
        ["hp_ts500",    "500 training timesteps",           "0.760", "0.0664"],
        ["hp_ema",      "EMA decay 0.9999",                 "0.760", "0.0643"],
        ["hp_cosine",   "cosine LR schedule",               "0.758", "0.0714"],
        ["hp_lr_low",   "lr = 5e-5",                        "0.753", "0.0685"],
    ],
    highlight_rows=[0],
    col_widths=[2.4, 4.6, 1.8, 1.8],
    filename="table2_hyperparameters.png",
)

# ── Table 3: Loss sweep ───────────────────────────────────────────────────────
render_table(
    title="Sweep 3 — Loss Functions",
    subtitle="Fixed: medium arch · lr = 1e-4 · 300 epochs | Green = best, Red = worst",
    columns=["Experiment", "Loss Description", "Best SSIM", "Best MAE"],
    rows=[
        ["loss_et_l1_w10",    "L1 + 10× enhancing tumour (ET) only",   "0.761", "0.0733"],
        ["loss_et_l1_w20",    "L1 + 20× enhancing tumour (ET) only",   "0.739", "0.0758"],
        ["loss_et_l1_w5",     "L1 + 5× enhancing tumour (ET) only",    "0.762", "0.0685"],
        ["loss_l1",           "Standard L1  ← BEST SSIM",              "0.769", "0.0668"],
        ["loss_l1ssim",       "0.5 · L1 + 0.5 · SSIM",                "0.757", "0.0721"],
        ["loss_l2",           "L2 (smoother gradients)",               "0.763", "0.0697"],
        ["loss_ncc",          "NCC — intensity-invariant  ← WORST",    "0.608", "0.0960"],
        ["loss_ssim",         "SSIM only",                             "0.747", "0.0736"],
        ["loss_tumor_l1_w10", "L1 + 10× whole-tumour weight",          "0.761", "0.0671"],
        ["loss_tumor_l1_w2",  "L1 + 2× whole-tumour weight",           "0.763", "0.0707"],
        ["loss_tumor_l1_w5",  "L1 + 5× whole-tumour weight  ← BEST MAE","0.767","0.0666"],
    ],
    highlight_rows=[3, 10],   # loss_l1 (best SSIM), loss_tumor_l1_w5 (best MAE)
    worst_rows=[6],           # loss_ncc
    col_widths=[2.8, 4.4, 1.8, 1.8],
    filename="table3_loss_functions.png",
)

# ── Table 4: Best config summary ─────────────────────────────────────────────
render_table(
    title="Overall Best Configuration",
    subtitle="Combining winners from all three sweeps",
    columns=["Setting", "Best Value", "Runner-up"],
    rows=[
        ["Architecture",      "rflow_large [64,128,256,512]  178.6M", "rflow_medium  76.5M"],
        ["Learning rate",     "2e-4  (hp_lr_high)",                   "1e-4  (baseline)"],
        ["Loss function",     "L1  (loss_l1)",                        "tumor_l1 w5 (tumour focus)"],
        ["Training timesteps","1000",                                  "250  (faster, -0.007 MAE)"],
        ["Attention levels",  "2  (default)",                         "3  (+0.001 SSIM)"],
        ["Res blocks/level",  "2  (default)",                         "3  (+0.003 SSIM)"],
    ],
    highlight_rows=[0, 1, 2],
    col_widths=[3.0, 4.6, 3.8],
    filename="table4_best_config.png",
)

print("\nAll tables saved to figures/tables/")
