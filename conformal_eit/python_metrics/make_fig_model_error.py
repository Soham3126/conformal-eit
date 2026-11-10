"""Model-error figure (pixel error bars versus model-error level) drawn from
conformal_eit/results/exp2_intervals.csv, with readable legend names.

    python make_fig_model_error.py            (writes results/exp2_model_error.png)
    python make_fig_model_error.py --alpha 0.10 --out C:/path/to/figure.png
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [("Bayes", "Bayes credible", "Bayes credible"), ("BAE", "BAE credible", "BAE credible"),
          ("Bayes", "CRC grp adaptive", "Bayes + conformal"), ("BAE", "CRC grp adaptive", "BAE + conformal"),
          ("GREIT", "CRC grp adaptive", "GREIT + conformal"), ("TV", "CRC grp adaptive", "TV + conformal")]
COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE"]
PANELS = [("miscov_active", "miscoverage, active pixels", "center"),
          ("width_active", "interval width, active pixels", "upper right"),
          ("miscov_true_anomaly", "miscoverage, true anomaly (not guaranteed)", "center right")]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..")))
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = os.path.join(args.root, "conformal_eit", "results")
    out = args.out or os.path.join(res, "exp2_model_error.png")
    with open(os.path.join(res, "exp2_intervals.csv")) as f:
        rows = [r for r in csv.DictReader(f) if abs(float(r["alpha"]) - args.alpha) < 1e-9]
    levels = sorted({int(r["level"]) for r in rows})

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 10, "mathtext.fontset": "cm"})
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for p, (col, ylabel, loc) in enumerate(PANELS):
        ax = axs[p]
        for (est, method, name), color in zip(SERIES, COLORS):
            s = sorted((r for r in rows if r["estimate"] == est and r["method"] == method), key=lambda r: int(r["level"]))
            if s:
                ax.plot([int(r["level"]) for r in s], [float(r[col]) for r in s], "-o", color=color, lw=1.4,
                        ms=6, mfc="none", mew=1.4, label=name, clip_on=False, zorder=3)
        if p == 0:
            ax.plot([levels[0], levels[-1]], [args.alpha] * 2, "k--", lw=0.8, label=r"target $\alpha$")
            ax.set_ylim(0, 1); ax.set_yticks([i / 10 for i in range(11)])
        elif p == 1:
            ax.set_ylim(0, 2.5)
        else:
            ax.set_ylim(0.3, 1.0)
        ax.set_xlim(levels[0], levels[-1]); ax.set_xticks(levels); ax.grid(True, color="#E0E0E0", lw=0.6)
        ax.set_xlabel("model-error level", fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(rf"$\alpha$ = {args.alpha:.2f}", fontweight="bold", fontsize=11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(direction="in", length=3)
        ax.legend(loc=loc, fontsize=7.5, framealpha=1, edgecolor="k", fancybox=False).get_frame().set_linewidth(0.6)
    fig.tight_layout(w_pad=3)
    fig.savefig(out, dpi=200, facecolor="w")
    print(f"> Saved {out}")
