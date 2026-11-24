"""Comparison of the conformal sets with three established EIT reconstructions.

Each established method reconstructs an image and is read with the standard
GREIT 25%-of-peak rule; the conformal inner / outer sets are calibrated on GREIT
(alpha = 0.10), as in the rest of the paper.

    Gauss-Newton (one step, NOSER prior)   x = (J'J + a diag(J'J))^-1 J'y      (Cheney et al. 1990)
    Tikhonov regularisation (identity)      x = (J'J + a I)^-1 J'y              (Tikhonov & Arsenin 1977)
    Total variation                         argmin 0.5/s^2 ||Jx-y||^2 + mu ||Dx||_1  (Chambolle-Pock solver)

Regularisation weights are tuned with the same rule as the paper's Bayesian and TV
estimates: lowest mean relative error on a separate tuning set (simulations:
conf_tune.mat; real tank: the first 60 simulated KTC phantoms in conf_ktc_sim.mat).
Simulated TV images are the ones cached by main_conformal.m (conf_tv_level<k>.mat).

Real voltages are put into model units with the scale implied by the noise level
printed by exp3_ktc_real.m (--ktc_sigma, default 0.0152); this matters only for TV.

    python run_comparison.py
Writes results/exp6_comparison_regions.csv (Table 5), results/exp6_comparison_metrics.csv
(extra columns of Table 4) and results/exp6_comparison_tables.tex.
"""
import argparse, os
import h5py
import numpy as np
import scipy.io as sio
from scipy import sparse
from run_seg_metrics import (CAL_FRAC, I25, LGRID, N_SPLITS, SEED, Grid, crc_index, load_ktc,
                             rel_score, seg_loss_curves)
from seg_metrics import METRICS, all_metrics

ALPHA = 0.10
RECON = ["gn", "tikhonov", "tv"]
LABEL = {"gn": "Gauss--Newton (25\\% rule)", "tikhonov": "Tikhonov (25\\% rule)", "tv": "Total variation (25\\% rule)",
         "conf_inner": "Conformal inner set (GREIT)", "conf_outer": "Conformal outer set (GREIT)"}
REGION_METRICS = ["spurious", "missed", "area", "dice", "iou", "sensitivity", "precision"]


# ---------- data ----------
def load_ops_full(path):
    with h5py.File(path, "r") as f:
        o = f["ops"]
        g = o["Dg"]; nrow = int(g.attrs["MATLAB_sparse"]); P = int(o["P"][()].squeeze())
        Dg = sparse.csc_matrix((g["data"][()], g["ir"][()], g["jc"][()]), shape=(nrow, P))
        return {"R": o["R"][()].T, "J": o["Jp"][()].T, "C": o["C"][()].T, "dx": float(o["dx"][()].squeeze()),
                "Dg": Dg.tocsr()}


def load_set(path):
    with h5py.File(path, "r") as f:
        D = f["D"]
        return {"Y": D["Y"][()].T, "XT": D["XT"][()].T, "T": D["T"][()].T.astype(bool),
                "sigma_m": float(D["sigma_m"][()].squeeze())}


# ---------- reconstructions ----------
def linear_operator(J, kind, a):
    JtJ = J.T @ J
    Reg = np.diag(np.diag(JtJ)) if kind == "gn" else np.eye(JtJ.shape[0]) * (np.trace(JtJ) / JtJ.shape[0])
    return np.linalg.solve(JtJ + a * Reg, J.T)


def rel_err(X, XT):
    return float(np.mean(np.linalg.norm(X - XT, axis=0) / np.maximum(np.linalg.norm(XT, axis=0), 1e-300)))


def tune_linear(J, kind, Ytune, XTtune, grid=np.logspace(-8, 2, 41)):
    errs = [rel_err(linear_operator(J, kind, a) @ Ytune, XTtune) for a in grid]
    k = int(np.argmin(errs))
    if k in (0, len(grid) - 1):
        print(f"   warning: {kind} weight at the edge of the grid")
    return linear_operator(J, kind, grid[k]), grid[k], errs[k]


def tv_recon(Y, J, Dg, sigma, mu, maxit=2000, tol=1e-4):
    """Port of recon/tv_recon.m (diagonally preconditioned Chambolle-Pock), all columns at once."""
    K = J / sigma; Kt = K.T; Dt = Dg.T.tocsr()
    absK = np.abs(K); absD = abs(Dg)
    tau = 1.0 / np.maximum(absK.sum(0) + np.asarray(absD.sum(0)).ravel(), np.finfo(float).eps)
    s1 = 1.0 / np.maximum(absK.sum(1), np.finfo(float).eps)
    s2 = 1.0 / np.maximum(np.asarray(absD.sum(1)).ravel(), np.finfo(float).eps)
    d = Y / sigma; N = Y.shape[1]; P = J.shape[1]
    x = np.zeros((P, N)); xb = x.copy(); y1 = np.zeros((K.shape[0], N)); y2 = np.zeros((Dg.shape[0], N))
    active = np.ones(N, bool)
    for it in range(1, maxit + 1):
        a = active
        y1[:, a] = (y1[:, a] + s1[:, None] * (K @ xb[:, a] - d[:, a])) / (1 + s1[:, None])
        y2[:, a] = np.clip(y2[:, a] + s2[:, None] * (Dg @ xb[:, a]), -mu, mu)
        xn = x[:, a] - tau[:, None] * (Kt @ y1[:, a] + Dt @ y2[:, a])
        xb[:, a] = 2 * xn - x[:, a]
        if it % 50 == 0:
            conv = np.linalg.norm(xn - x[:, a], axis=0) <= tol * np.maximum(np.linalg.norm(xn, axis=0), np.finfo(float).eps)
            x[:, a] = xn
            idx = np.where(a)[0]; active[idx[conv]] = False
            if not active.any():
                break
        else:
            x[:, a] = xn
    return x


def tune_tv(J, Dg, sigma, Ytune, XTtune, grid=2.0 ** np.arange(-6, 15, 2)):
    errs = []
    for mu in grid:
        errs.append(rel_err(tv_recon(Ytune, J, Dg, sigma, mu), XTtune))
        print(f"      TV mu = {mu:<8g} relative error {errs[-1]:.3f}")
    k = int(np.argmin(errs))
    return grid[k], errs[k]


# ---------- scoring ----------
def region_scores(mask, t):
    tp = float(np.sum(mask & t)); fp = float(np.sum(mask & ~t)); fn = float(np.sum(~mask & t)); nt = max(tp + fn, 1.0)
    return np.array([min(fp / nt, 1.0), fn / nt, (tp + fp) / nt, 2 * tp / max(2 * tp + fp + fn, 1.0),
                     tp / max(tp + fp + fn, 1.0), tp / nt, tp / (tp + fp) if tp + fp > 0 else np.nan])


def evaluate(images, Sg, T, grid, splits):
    """images: {method: P x N reconstruction}; Sg: GREIT relative score for the conformal sets."""
    out = {}
    masks = {m: rel_score(X) >= LGRID[I25] for m, X in images.items()}
    for m, M in masks.items():
        reg = np.array([region_scores(M[:, i], T[:, i]) for i in range(T.shape[1])])
        met = np.array([all_metrics(M[:, i], T[:, i], grid) for i in range(T.shape[1])])
        out[m] = (np.mean([np.nanmean(reg[t], 0) for _, t in splits], 0),
                  np.mean([np.nanmean(met[t], 0) for _, t in splits], 0), 0.25)
    Lin, Lout = seg_loss_curves(Sg, T); acc = {"conf_inner": [], "conf_outer": []}; lam = {"conf_inner": [], "conf_outer": []}
    for cal, tst in splits:
        for m, k in (("conf_inner", crc_index(Lin[cal], ALPHA, "nonincreasing")),
                     ("conf_outer", crc_index(Lout[cal], ALPHA, "nondecreasing"))):
            acc[m].append(np.nanmean([region_scores(Sg[:, i] >= LGRID[k], T[:, i]) for i in tst], 0)); lam[m].append(LGRID[k])
    for m in acc:
        out[m] = (np.mean(acc[m], 0), None, float(np.mean(lam[m])))
    return out


def run(root, ktc_sigma):
    data = os.path.join(root, "conformal_eit", "data"); results = {}
    # ---- simulations, level 3 ----
    ops = load_ops_full(os.path.join(data, "conf_ops_b3cr.mat")); grid = Grid(ops["C"], ops["dx"])
    tune = load_set(os.path.join(data, "conf_tune.mat"))
    G_gn, a_gn, e_gn = tune_linear(ops["J"], "gn", tune["Y"], tune["XT"])
    G_tk, a_tk, e_tk = tune_linear(ops["J"], "tikhonov", tune["Y"], tune["XT"])
    print(f"   simulations: Gauss-Newton weight {a_gn:.3g} (tuning error {e_gn:.3f}), Tikhonov weight {a_tk:.3g} ({e_tk:.3f})")
    lv = 3; L = lv + 1
    D = load_set(os.path.join(data, f"conf_level{lv}.mat"))
    Xtv = sio.loadmat(os.path.join(data, f"conf_tv_level{lv}.mat"))["Xt"]
    n = D["Y"].shape[1]; nc = int(round(CAL_FRAC * n)); rng = np.random.default_rng(SEED + L)
    splits = [(p[:nc], p[nc:]) for p in (rng.permutation(n) for _ in range(N_SPLITS))]
    results["sim_level3"] = evaluate({"gn": G_gn @ D["Y"], "tikhonov": G_tk @ D["Y"], "tv": Xtv},
                                     rel_score(ops["R"] @ D["Y"]), D["T"], grid, splits)
    print(f"   simulated level 3: {n} phantoms")
    # ---- real KTC2023 ----
    K = load_ktc(root)
    if K is None:
        print("   (real data not found, skipped)"); return results
    opk = load_ops_full(os.path.join(data, "conf_ops_ktc.mat"))
    ks = load_set(os.path.join(data, "conf_ktc_sim.mat"))
    Yt, XTt = ks["Y"][:, :60], ks["XT"][:, :60]
    G_gn, a_gn, e_gn = tune_linear(opk["J"], "gn", Yt, XTt)
    G_tk, a_tk, e_tk = tune_linear(opk["J"], "tikhonov", Yt, XTt)
    print(f"   KTC: Gauss-Newton weight {a_gn:.3g} ({e_gn:.3f}), Tikhonov weight {a_tk:.3g} ({e_tk:.3f}); tuning TV ...")
    mu, e_mu = tune_tv(opk["J"], opk["Dg"], ks["sigma_m"], Yt, XTt)
    # real voltages -> model units: same out-of-range noise estimate as exp3_ktc_real.m
    U, sv, _ = np.linalg.svd(opk["J"], full_matrices=False); r = int(np.sum(sv > sv[0] * 1e-8)); U = U[:, :r]
    Yr = K["Y"]; res = Yr - U @ (U.T @ Yr)
    sig_raw = float(np.median(np.sqrt(np.sum(res ** 2, 0) / max(Yr.shape[0] - r, 1))))
    Ym = Yr * (ktc_sigma / sig_raw)
    print(f"   KTC: TV weight {mu:g} ({e_mu:.3f}); voltage scale {sig_raw / ktc_sigma:.4g}")
    Xtv = tv_recon(Ym, opk["J"], opk["Dg"], ktc_sigma, mu)
    n = Ym.shape[1]
    splits = [(np.array([i for i in range(n) if i != j]), np.array([j])) for j in range(n)]
    results["ktc_real"] = evaluate({"gn": G_gn @ Ym, "tikhonov": G_tk @ Ym, "tv": Xtv},
                                   rel_score(K["X"]), K["T"], K["grid"], splits)
    print(f"   real KTC2023: {n} phantoms (leave-one-out)")
    return results


def write(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    order = RECON + ["conf_inner", "conf_outer"]
    with open(os.path.join(out_dir, "exp6_comparison_regions.csv"), "w") as f:
        f.write("dataset,method,threshold," + ",".join(REGION_METRICS) + "\n")
        for ds, r in results.items():
            for m in order:
                f.write(f"{ds},{m},{r[m][2]:.4f}," + ",".join(f"{v:.6f}" for v in r[m][0]) + "\n")
    with open(os.path.join(out_dir, "exp6_comparison_metrics.csv"), "w") as f:
        f.write("dataset,method," + ",".join(METRICS) + "\n")
        for ds, r in results.items():
            for m in RECON:
                f.write(f"{ds},{m}," + ",".join(f"{v:.6f}" for v in r[m][1]) + "\n")
    with open(os.path.join(out_dir, "exp6_comparison_tables.tex"), "w") as f:
        for ds, r in results.items():
            f.write(f"% {ds}\n")
            for m in order:
                v = r[m][0]
                f.write(f"{LABEL[m]} & {r[m][2]:.2f} & {v[0]:.3f} & {v[1]:.3f} & {v[2]:.2f} & {v[3]:.3f} & {v[6]:.3f}\\\\\n")
            f.write("\n")
    for ds, r in results.items():
        print(f"\n{ds}: anomaly regions, alpha = {ALPHA:.2f}")
        print(f"{'method':30s}{'thr':>6s}" + "".join(f"{k:>12s}" for k in REGION_METRICS))
        for m in order:
            print(f"{LABEL[m].replace(chr(92), '').replace('--', '-'):30s}{r[m][2]:6.2f}" + "".join(f"{v:12.3f}" for v in r[m][0]))
        print(f"{'segmentation metrics':30s}" + "".join(f"{k[:10]:>11s}" for k in METRICS))
        for m in RECON:
            print(f"{m:30s}" + "".join(f"{v:11.3f}" for v in r[m][1]))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ktc_sigma", type=float, default=0.0152,
                    help="real-data noise std in model units, as printed by exp3_ktc_real.m")
    a = ap.parse_args()
    print("> Experiment 6: comparison with Gauss-Newton, Tikhonov and TV")
    write(run(a.root, a.ktc_sigma), a.out or os.path.join(a.root, "conformal_eit", "results"))
