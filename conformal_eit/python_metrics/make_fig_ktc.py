"""Real-phantom figure: (photograph,) ground truth, |GREIT| and the conformal
inner / outer sets for four KTC2023 phantoms, calibrated leave-one-out.

    python make_fig_ktc.py
    python make_fig_ktc.py --photos C:/path/to/photos --out C:/path/to/figure.png

Photographs are optional. If the folder (default: <root>/paper/figures/ktc_photos)
holds ktc_photo_<name>.jpg or .png for every shown phantom, a photograph row is
added on top and the figure is written to <root>/paper/figures/fig_ktc_examples.png.
Otherwise a three-row figure is written to conformal_eit/results/fig_ktc_examples.png.
The photographs are not part of the KTC2023 dataset; they come from the KTC2023
paper (Rasanen et al., Appl. Math. Mod. Chall. 2(2), 2024) and are not distributed here.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from PIL import Image, ImageOps
from run_seg_metrics import LGRID, crc_index, load_ktc, rel_score, seg_loss_curves

C_BG, C_TRUE, C_MISS = (0.88, 0.88, 0.88), (0.85, 0.33, 0.10), (0.75, 0.10, 0.10)
C_OUTER, C_INNER = (0.60, 0.78, 0.95), (0.08, 0.30, 0.65)
# blue-green-yellow scale matching the one MATLAB uses for the GREIT images
GREIT_CMAP = LinearSegmentedColormap.from_list("greit", [
    (0.248, 0.167, 0.708), (0.266, 0.216, 0.822), (0.273, 0.308, 0.935), (0.246, 0.410, 0.966),
    (0.197, 0.491, 0.970), (0.164, 0.564, 0.928), (0.133, 0.632, 0.888), (0.061, 0.705, 0.809),
    (0.090, 0.746, 0.707), (0.246, 0.776, 0.581), (0.354, 0.798, 0.463), (0.601, 0.788, 0.281),
    (0.802, 0.754, 0.168), (0.951, 0.737, 0.213), (0.978, 0.824, 0.176), (0.961, 0.891, 0.150),
    (0.970, 0.963, 0.097)])


def tank_image(grid, v):
    """n x n image with NaN outside the tank, row 0 at the bottom (use origin='lower')."""
    img = np.full((grid.n, grid.n), np.nan)
    img[grid.iy, grid.ix] = v
    return img


def find_photo(folder, name):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(folder, f"ktc_photo_{name}{ext}")
        if os.path.exists(p):
            return p
    return None


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..")))
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--show", nargs="+", default=["data1", "data17", "data24", "data9"])
    ap.add_argument("--photos", default=None, help="folder with ktc_photo_<name>.jpg/.png")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    K = load_ktc(args.root)
    if K is None:
        raise SystemExit("KTC data not found: run run_exp3.m in MATLAB first.")
    grid, names, T = K["grid"], K["names"], K["T"]
    S = rel_score(K["X"]); n = S.shape[1]
    Lin, Lout = seg_loss_curves(S, T)

    pdir = args.photos or os.path.join(args.root, "paper", "figures", "ktc_photos")
    photos = [find_photo(pdir, nm) for nm in args.show]
    with_photos = all(photos)
    out = args.out or (os.path.join(args.root, "paper", "figures", "fig_ktc_examples.png") if with_photos
                       else os.path.join(args.root, "conformal_eit", "results", "fig_ktc_examples.png"))

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "mathtext.fontset": "dejavusans"})
    nrow, ncol = (4 if with_photos else 3), len(args.show)
    fig, axs = plt.subplots(nrow, ncol, figsize=(1.85 * ncol, 2.0 * nrow), squeeze=False)
    tkw = dict(fontsize=8.5, fontweight="bold", pad=4)
    for c, nm in enumerate(args.show):
        j = names.index(nm)
        cal = np.array([i for i in range(n) if i != j])            # leave-one-out
        ki = crc_index(Lin[cal], args.alpha, "nonincreasing")
        ko = crc_index(Lout[cal], args.alpha, "nondecreasing")
        r = 0
        if with_photos:
            ph = Image.open(photos[c]).convert("RGB")
            if K["mirror"]:                                        # same mirror as the truth registration
                ph = ImageOps.mirror(ph)
            axs[r, c].imshow(np.asarray(ph), interpolation="lanczos")
            axs[r, c].set_title(f"photograph ({nm})", **tkw); r += 1
        axs[r, c].imshow(tank_image(grid, T[:, j].astype(float)), origin="lower", vmin=0, vmax=1,
                         cmap=ListedColormap([C_BG, C_TRUE]), interpolation="nearest")
        axs[r, c].set_title(f"truth ({nm})", **tkw); r += 1
        axs[r, c].imshow(tank_image(grid, np.abs(K["X"][:, j])), origin="lower", cmap=GREIT_CMAP,
                         interpolation="nearest")
        axs[r, c].set_title("|GREIT|", **tkw); r += 1
        m = (S[:, j] >= LGRID[ko]).astype(float) + (S[:, j] >= LGRID[ki])   # 1 outer only, 2 inner
        m[T[:, j] & (m == 0)] = -1                                          # missed anomaly pixels
        axs[r, c].imshow(tank_image(grid, m), origin="lower", vmin=-1.5, vmax=2.5,
                         cmap=ListedColormap([C_MISS, C_BG, C_OUTER, C_INNER]), interpolation="nearest")
        axs[r, c].set_title(rf"$\lambda_{{in}}$={LGRID[ki]:.2f}, $\lambda_{{out}}$={LGRID[ko]:.2f}", **tkw)
    for ax in axs.ravel():
        ax.axis("off")
    labels = {"true anomaly": C_TRUE, "inner set": C_INNER, "outer set only": C_OUTER,
              "missed anomaly": C_MISS, "tank (no anomaly)": C_BG}
    fig.legend(handles=[Patch(facecolor=v, edgecolor=(0.4, 0.4, 0.4), label=k) for k, v in labels.items()],
               loc="lower center", ncol=5, frameon=False, fontsize=8.5, handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - 0.28 / (2.0 * nrow), bottom=0.44 / (2.0 * nrow),
                        wspace=0.12, hspace=0.20)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="w")
    print(f"> Saved {out} ({'with' if with_photos else 'without'} photograph row)")
