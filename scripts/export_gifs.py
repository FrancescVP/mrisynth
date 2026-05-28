"""Generate axial-slice GIFs for each experiment sweep group.

Layout: rows = experiments (GT first), columns = subjects.
Each frame = one axial slice scrolling through all brains simultaneously.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont

PRED_DIR = Path("predictions")
OUT_DIR  = Path("figures/gifs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All 5 val cases present in every experiment
CASES = [
    "BraTS-MEN-00008-000",
    "BraTS-MEN-00031-000",
    "BraTS-MEN-00033-000",
    "BraTS-MEN-00043-000",
    "BraTS-MEN-00059-000",
]

CELL_W   = 160    # pixels per brain slice (width)
CELL_H   = 128    # pixels per brain slice (height)
LABEL_H  = 22    # label strip height below each cell
PAD      = 4     # gap between cells
COLS     = 4     # experiments per row
BG       = 10    # background grey value
FPS      = 6     # frames per second (slow)
Z_STEP   = 2     # sample every N axial slices

try:
    FONT     = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    FONT_SM  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except OSError:
    FONT = FONT_SM = ImageFont.load_default()

SWEEP_GROUPS: dict[str, list[str]] = {
    "arch": ["rflow_tiny", "rflow_small", "rflow_medium", "rflow_deep", "rflow_large"],
    "hp":   ["hp_baseline", "hp_cosine", "hp_decay", "hp_ts500", "hp_ts250",
             "hp_lr_high", "hp_lr_low", "hp_attn3", "hp_resblk3", "hp_ema"],
    "loss": [
        "loss_l1",            # SSIM 0.769 ★
        "loss_tumor_l1_w5",   # SSIM 0.767
        "loss_tumor_l1_w2",   # SSIM 0.763
        "loss_l2",            # SSIM 0.763
        "loss_et_l1_w5",      # SSIM 0.762
        "loss_tumor_l1_w10",  # SSIM 0.761
        "loss_et_l1_w10",     # SSIM 0.761
        "loss_l1ssim",        # SSIM 0.757
        "loss_ssim",          # SSIM 0.747
        "loss_et_l1_w20",     # SSIM 0.739
        "loss_ncc",           # SSIM 0.608 ✗
    ],
}

LABEL_OVERRIDE = {
    "gt":                "Ground Truth",
    "rflow_tiny":        "Tiny  19M",
    "rflow_small":       "Small  45M",
    "rflow_medium":      "Medium  77M",
    "rflow_deep":        "Deep  76M",
    "rflow_large":       "Large  179M ★",
    "hp_baseline":       "Baseline lr=1e-4",
    "hp_cosine":         "Cosine LR",
    "hp_decay":          "LR Decay",
    "hp_ts500":          "500 Timesteps",
    "hp_ts250":          "250 Timesteps",
    "hp_lr_high":        "LR=2e-4 ★",
    "hp_lr_low":         "LR=5e-5",
    "hp_attn3":          "3 Attn Levels",
    "hp_resblk3":        "3 Res Blocks",
    "hp_ema":            "EMA 0.9999",
    "loss_l1":           "L1 ★",
    "loss_l2":           "L2",
    "loss_ssim":         "SSIM",
    "loss_ncc":          "NCC ✗",
    "loss_l1ssim":       "L1+SSIM",
    "loss_tumor_l1_w2":  "Tumor L1 w=2",
    "loss_tumor_l1_w5":  "Tumor L1 w=5",
    "loss_tumor_l1_w10": "Tumor L1 w=10",
    "loss_et_l1_w5":     "ET L1 w=5",
    "loss_et_l1_w10":    "ET L1 w=10",
    "loss_et_l1_w20":    "ET L1 w=20",
}


def load_norm(path: Path) -> np.ndarray:
    vol = nib.load(path).get_fdata(dtype=np.float32)
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99.5)
    vol = np.clip(vol, lo, hi)
    if hi > lo:
        vol = (vol - lo) / (hi - lo)
    return vol


def axial_slice(vol: np.ndarray, z: int) -> np.ndarray:
    """Return uint8 H×W axial slice at z, rotated 180° then 45° clockwise."""
    sl = vol[:, :, z]           # (128, 160)
    sl = np.rot90(sl, k=2)      # 180° base orientation
    sl = (sl * 255).astype(np.uint8)
    img = Image.fromarray(sl, mode="L")
    img = img.rotate(90, expand=True, fillcolor=0)    # 90° counter-clockwise
    img = img.resize((CELL_W, CELL_H), Image.LANCZOS)
    return np.array(img)


def make_frame(panels: list[tuple[str, np.ndarray]], z: int) -> Image.Image:
    """panels = [(label, vol), ...] — GT first, then experiments."""
    n = len(panels)
    n_cols = COLS
    n_rows = (n + n_cols - 1) // n_cols
    W = n_cols * (CELL_W + PAD) - PAD
    H = n_rows * (CELL_H + LABEL_H + PAD) - PAD
    frame = Image.new("RGB", (W, H), color=(BG, BG, BG))
    draw  = ImageDraw.Draw(frame)

    for idx, (label, vol) in enumerate(panels):
        row, col = divmod(idx, n_cols)
        x = col * (CELL_W + PAD)
        y = row * (CELL_H + LABEL_H + PAD)

        sl = axial_slice(vol, z)
        frame.paste(Image.fromarray(sl).convert("RGB"), (x, y))

        # Label colours
        if label == "Ground Truth":
            lbl_col, lbl_bg = (255, 215, 0), (50, 40, 0)
        elif "★" in label:
            lbl_col, lbl_bg = (100, 230, 100), (0, 40, 0)
        elif "✗" in label:
            lbl_col, lbl_bg = (230, 80, 80), (40, 0, 0)
        else:
            lbl_col, lbl_bg = (200, 200, 200), (30, 30, 30)

        draw.rectangle([x, y + CELL_H, x + CELL_W, y + CELL_H + LABEL_H], fill=lbl_bg)
        draw.text((x + CELL_W // 2, y + CELL_H + LABEL_H // 2),
                  label, font=FONT_SM, fill=lbl_col, anchor="mm")

    return frame


def make_gif(group_name: str, exps: list[str]):
    for case in CASES:
        print(f"\n{'='*52}")
        print(f"  {group_name} | {case}")

        # Ground truth
        gt_path = None
        for exp in exps:
            p = PRED_DIR / exp / case / "gt_t1ce.nii.gz"
            if p.exists():
                gt_path = p
                break
        if gt_path is None:
            print(f"  [!] No GT found, skipping")
            continue

        panels: list[tuple[str, np.ndarray]] = [("Ground Truth", load_norm(gt_path))]

        for exp in exps:
            pred_path = PRED_DIR / exp / case / "pred_t1ce.nii.gz"
            if not pred_path.exists():
                print(f"  [!] Missing {pred_path}")
                continue
            panels.append((LABEL_OVERRIDE.get(exp, exp), load_norm(pred_path)))
            print(f"  Loaded {exp}")

        n_z = min(vol.shape[2] for _, vol in panels)
        z_slices = list(range(0, n_z, Z_STEP))

        frames = [make_frame(panels, z) for z in z_slices]

        case_dir = OUT_DIR / case
        case_dir.mkdir(parents=True, exist_ok=True)
        out_path = case_dir / f"sweep_{group_name}.gif"
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / FPS),
            loop=0,
            optimize=False,
        )
        print(f"  Saved → {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    for group, exps in SWEEP_GROUPS.items():
        make_gif(group, exps)
    print("\nAll GIFs done →", OUT_DIR)
