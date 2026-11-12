"""Comparison of the conformal method with four established alternatives.

Regions (simulated levels 0-3 and the real KTC2023 phantoms):
    GREIT 25% rule        fixed threshold at 25% of the peak (GREIT convention)
    Otsu                  per-image automatic threshold (Otsu, 1979); no calibration data
    Dice-tuned threshold  single threshold that maximises mean Dice on the calibration phantoms (no guarantee)
    Conformal inner/outer conformal risk control, alpha = 0.10
Error bars (simulated levels 0-3; real tank has no true conductivity values):
    Bayesian credible intervals, Bayesian approximation error (BAE), conformal around GREIT
    -> read from results/exp2_intervals.csv produced by main_conformal.m

Same data, splits and protocol as run_seg_metrics.py (200 random 50/50 splits for
simulations, leave-one-out for the 25 real phantoms).

    python run_comparison.py
Writes results/exp6_comparison_regions.csv, results/exp6_comparison_errorbars.csv
and results/exp6_comparison_tables.tex
"""
import argparse, csv, os
import h5py
import numpy as np
from skimage.filters import threshold_otsu
from run_seg_metrics import (CAL_FRAC, I25, LGRID, N_SPLITS, SEED, crc_index, load_ktc, load_ops,
                             rel_score, seg_loss_curves)

ALPHA = 0.10
REGION_METHODS = ["greit25", "otsu", "dice_tuned", "conf_inner", "conf_outer"]
REGION_LABEL = {"greit25": "GREIT 25\\% rule", "otsu": "Otsu threshold", "dice_tuned": "Dice-tuned threshold",
                "conf_inner": "Conformal inner set", "conf_outer": "Conformal outer set"}
REGION_METRICS = ["spurious", "missed", "area", "dice", "iou", "sensitivity", "precision"]


def counts(mask, t):
    tp = float(np.sum(mask & t)); fp = float(np.sum(mask & ~t)); fn = float(np.sum(~mask & t))
    return tp, fp, fn


def region_scores(mask, t):
    """spurious area / true area (capped at 1, as in the CRC loss), missed fraction, area ratio,
    Dice, IoU, sensitivity, precision."""
    tp, fp, fn = counts(mask, t); nt = max(tp + fn, 1.0)
    return np.array([min(fp / nt, 1.0), fn / nt, (tp + fp) / nt,
                     2 * tp / max(2 * tp + fp + fn, 1.0), tp / max(tp + fp + fn, 1.0),
                     tp / nt, tp / (tp + fp) if tp + fp > 0 else np.nan])


def dice_curves(S, T):
    """D[i,k] = Dice of {S[:,i] >= LGRID[k]} against T[:,i]."""
    N = S.shape[1]; D = np.zeros((N, len(LGRID)))
    for i in range(N):
        inside = S[:, i][:, None] >= LGRID[None, :]; t = T[:, i][:, None]
        tp = (inside & t).sum(0); fp = (inside & ~t).sum(0); fn = (~inside & t).sum(0)
        D[i] = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return D


class Scorer:
    """Caches region scores per phantom for grid thresholds and for Otsu."""

    def __init__(self, S, T):
        self.S, self.T, self.grid_cache = S, T, {}
        self.otsu = np.array([region_scores(S[:, i] >= threshold_otsu(S[:, i]), T[:, i]) for i in range(S.shape[1])])

    def at(self, i, k):
        if (i, k) not in self.grid_cache:
            self.grid_cache[(i, k)] = region_scores(self.S[:, i] >= LGRID[k], self.T[:, i])
        return self.grid_cache[(i, k)]


def evaluate(S, T, splits):
    """splits: list of (cal_idx, test_idx). Returns {method: mean scores} averaged over test
    phantoms, then over splits."""
    Lin, Lout = seg_loss_curves(S, T); D = dice_curves(S, T); sc = Scorer(S, T)
    acc = {m: [] for m in REGION_METHODS}; lam = {m: [] for m in ["dice_tuned", "conf_inner", "conf_outer"]}
    for cal, tst in splits:
        ki = crc_index(Lin[cal], ALPHA, "nonincreasing"); ko = crc_index(Lout[cal], ALPHA, "nondecreasing")
        kd = int(np.argmax(D[cal].mean(0)))
        lam["dice_tuned"].append(LGRID[kd]); lam["conf_inner"].append(LGRID[ki]); lam["conf_outer"].append(LGRID[ko])
        for m, rows in [("greit25", [sc.at(i, I25) for i in tst]), ("otsu", [sc.otsu[i] for i in tst]),
                        ("dice_tuned", [sc.at(i, kd) for i in tst]), ("conf_inner", [sc.at(i, ki) for i in tst]),
                        ("conf_outer", [sc.at(i, ko) for i in tst])]:
            acc[m].append(np.nanmean(rows, axis=0))
    out = {m: np.mean(acc[m], axis=0) for m in REGION_METHODS}
    thr = {"greit25": 0.25, "otsu": float(np.mean([threshold_otsu(S[:, i]) for i in range(S.shape[1])]))}
    thr.update({m: float(np.mean(v)) for m, v in lam.items()})
    return out, thr


def run_regions(root):
    data = os.path.join(root, "conformal_eit", "data"); R, _, _ = load_ops(os.path.join(data, "conf_ops_b3cr.mat"))
    results = {}
    for L, lv in enumerate(range(4), start=1):
        fn = os.path.join(data, f"conf_level{lv}.mat")
        if not os.path.exists(fn):
            print(f"   (level {lv} not found, skipped)"); continue
        with h5py.File(fn, "r") as f:
            Y, T = f["D"]["Y"][()].T, f["D"]["T"][()].T.astype(bool)
        S = rel_score(R @ Y); n = S.shape[1]; nc = int(round(CAL_FRAC * n))
        rng = np.random.default_rng(SEED + L)
        splits = [(p[:nc], p[nc:]) for p in (rng.permutation(n) for _ in range(N_SPLITS))]
        results[f"sim_level{lv}"] = evaluate(S, T, splits)
        print(f"   simulated level {lv}: {n} phantoms")
    K = load_ktc(root)
    if K is None:
        print("   (real data not found, skipped)")
    else:
        S = rel_score(K["X"]); n = S.shape[1]
        splits = [(np.array([i for i in range(n) if i != j]), np.array([j])) for j in range(n)]
        results["ktc_real"] = evaluate(S, K["T"], splits)
        print(f"   real KTC2023: {n} phantoms (leave-one-out)")
    return results


def read_errorbars(root):
    fn = os.path.join(root, "conformal_eit", "results", "exp2_intervals.csv")
    want = [("Bayes", "Bayes credible", "Bayesian credible intervals"), ("BAE", "BAE credible", "Bayesian approximation error"),
            ("GREIT", "CRC grp adaptive", "Conformal (GREIT)")]
    out = []
    with open(fn) as f:
        for r in csv.DictReader(f):
            if abs(float(r["alpha"]) - ALPHA) > 1e-9:
                continue
            for est, meth, label in want:
                if r["estimate"] == est and r["method"] == meth:
                    out.append((int(r["level"]), label, float(r["miscov_all"]), float(r["miscov_active"]),
                                float(r["miscov_background"]), float(r["width_active"])))
    return sorted(out, key=lambda x: (x[0], [w[2] for w in want].index(x[1])))


def write(root, out_dir, regions, ebars):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp6_comparison_regions.csv"), "w") as f:
        f.write("dataset,method,mean_threshold," + ",".join(REGION_METRICS) + "\n")
        for ds, (vals, thr) in regions.items():
            for m in REGION_METHODS:
                f.write(f"{ds},{m},{thr[m]:.4f}," + ",".join(f"{v:.6f}" for v in vals[m]) + "\n")
    with open(os.path.join(out_dir, "exp6_comparison_errorbars.csv"), "w") as f:
        f.write("level,method,miscov_all,miscov_active,miscov_background,width_active\n")
        for r in ebars:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f}\n")
    lines = []
    for ds, (vals, thr) in regions.items():
        print(f"\n{ds}: anomaly regions, alpha = {ALPHA:.2f}")
        print(f"{'method':24s}{'thr':>6s}" + "".join(f"{k:>12s}" for k in REGION_METRICS))
        for m in REGION_METHODS:
            print(f"{REGION_LABEL[m].replace(chr(92), ''):24s}{thr[m]:6.2f}" + "".join(f"{v:12.3f}" for v in vals[m]))
            lines.append((ds, m, thr[m], vals[m]))
    print(f"\npixel error bars (simulations), alpha = {ALPHA:.2f}")
    print(f"{'lvl':4s}{'method':32s}{'miscov all':>12s}{'active':>10s}{'background':>12s}{'width':>8s}")
    for r in ebars:
        print(f"{r[0]:<4d}{r[1]:32s}{r[2]:12.3f}{r[3]:10.3f}{r[4]:12.3f}{r[5]:8.2f}")
    with open(os.path.join(out_dir, "exp6_comparison_tables.tex"), "w") as f:
        for ds in regions:
            f.write(f"% {ds}\n\\begin{{tabular}}{{lcccccc}}\n\\toprule\n"
                    "Method & Threshold & Spurious & Missed & Area & Dice & Precision\\\\\n\\midrule\n")
            for d, m, t, v in lines:
                if d == ds:
                    f.write(f"{REGION_LABEL[m]} & {t:.2f} & {v[0]:.3f} & {v[1]:.3f} & {v[2]:.2f} & {v[3]:.3f} & {v[6]:.3f}\\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\n")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..")))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print("> Experiment 6: comparison with established methods")
    reg = run_regions(a.root); eb = read_errorbars(a.root)
    write(a.root, a.out or os.path.join(a.root, "conformal_eit", "results"), reg, eb)
