"""Experiment 5: standard segmentation metrics for the GREIT 25% rule and the
conformal inner / outer sets, on the simulated levels and the real KTC2023 phantoms.

Reads the cached MATLAB files produced by main_conformal.m (nothing is re-simulated):
    conformal_eit/data/conf_ops_b3cr.mat, conf_level<k>.mat          (simulations)
    conformal_eit/data/conf_ops_ktc.mat, data/ktc/*.mat,
    conformal_eit/results/exp3_ktc_registration.mat                 (real tank data)

Run from anywhere:
    python run_seg_metrics.py                     (project root = two folders above this file)
    python run_seg_metrics.py --root C:/path/to/EIT_Hybrid_Project

Writes to conformal_eit/results/:
    exp5_seg_metrics.csv          every dataset x alpha x method x metric
    exp5_seg_metrics_table.tex    LaTeX table at the chosen alpha (default 0.10)
"""
import argparse, glob, os, re
import h5py
import numpy as np
import scipy.io as sio
from seg_metrics import METRICS, Grid, all_metrics

SEED, N_SPLITS, CAL_FRAC, GRID_K = 2026, 200, 0.5, 401     # as in cfg_default.m
ALPHAS = [0.05, 0.10, 0.20]
LGRID = np.linspace(0.0, 1.0001, GRID_K)
I25 = int(np.argmin(np.abs(LGRID - 0.25)))                # the Graz 25% rule on this grid


# ---------- conformal risk control (ports of the MATLAB functions) ----------
def rel_score(X):
    return np.abs(X) / np.maximum(np.abs(X).max(axis=0, keepdims=True), np.finfo(float).eps)


def seg_loss_curves(S, T):
    """Lin[i,k] spurious area / true area (capped at 1), Lout[i,k] missed fraction."""
    N = S.shape[1]
    Lin, Lout = np.zeros((N, GRID_K)), np.zeros((N, GRID_K))
    for i in range(N):
        inside = S[:, i][:, None] >= LGRID[None, :]
        t = T[:, i][:, None]; nt = max(int(T[:, i].sum()), 1)
        Lin[i] = np.minimum((inside & ~t).sum(0) / nt, 1.0)
        Lout[i] = (~inside & t).sum(0) / nt
    return Lin, Lout


def crc_index(Lcal, alpha, direction, B=1.0):
    n = Lcal.shape[0]
    ok = np.where((n * Lcal.mean(0) + B) / (n + 1) <= alpha)[0]
    if direction == "nonincreasing":
        return int(ok[0]) if ok.size else GRID_K - 1
    return int(ok[-1]) if ok.size else 0


# ---------- metric evaluation with a cache over (phantom, threshold) ----------
class Evaluator:
    def __init__(self, S, T, grid):
        self.S, self.T, self.grid, self.cache = S, T, grid, {}

    def __call__(self, i, k):
        if (i, k) not in self.cache:
            self.cache[(i, k)] = all_metrics(self.S[:, i] >= LGRID[k], self.T[:, i], self.grid)
        return self.cache[(i, k)]

    def mean(self, idx, k):
        return np.nanmean(np.vstack([self(i, k) for i in idx]), axis=0)


def add_rows(rows, dataset, alpha, method, vals, extra=None):
    for name, v in zip(METRICS, vals):
        rows.append((dataset, alpha, method, name, float(v)))
    for name, v in (extra or {}).items():
        rows.append((dataset, alpha, method, name, float(v)))


# ---------- loaders ----------
def load_ops(path):
    with h5py.File(path, "r") as f:
        o = f["ops"]
        return o["R"][()].T, o["C"][()].T, float(o["dx"][()].squeeze())


def gt_to_pixels(G, C, ang_deg, flip):
    """Port of gt_to_pixels.m: sample the ground-truth image at the pixel centres."""
    x, y = C[:, 0].copy(), C[:, 1].copy()
    if flip:
        x = -x
    a = np.deg2rad(ang_deg)
    xr, yr = np.cos(a) * x - np.sin(a) * y, np.sin(a) * x + np.cos(a) * y
    nr, nc = G.shape
    col = np.clip(np.floor((xr + 1) / 2 * (nc - 1) + 0.5).astype(int), 0, nc - 1)
    row = np.clip(np.floor((1 - yr) / 2 * (nr - 1) + 0.5).astype(int), 0, nr - 1)
    return G[row, col] > 0


# ---------- experiments ----------
def run_simulated(root, rows):
    data = os.path.join(root, "conformal_eit", "data")
    R, C, dx = load_ops(os.path.join(data, "conf_ops_b3cr.mat"))
    grid = Grid(C, dx)
    for L, lv in enumerate(range(4), start=1):
        fn = os.path.join(data, f"conf_level{lv}.mat")
        if not os.path.exists(fn):
            print(f"   (level {lv}: {fn} not found, skipped)"); continue
        with h5py.File(fn, "r") as f:
            Y, T = f["D"]["Y"][()].T, f["D"]["T"][()].T.astype(bool)
        S = rel_score(R @ Y); n = S.shape[1]
        Lin, Lout = seg_loss_curves(S, T); ev = Evaluator(S, T, grid)
        ds = f"sim_level{lv}"
        add_rows(rows, ds, np.nan, "greit25", ev.mean(range(n), I25),
                 {"crc_spurious": Lin[:, I25].mean(), "crc_missed": Lout[:, I25].mean()})
        rng = np.random.default_rng(SEED + L); nc = int(round(CAL_FRAC * n))
        splits = [rng.permutation(n) for _ in range(N_SPLITS)]
        for a in ALPHAS:
            acc = {"inner": [], "outer": []}; risk = {"inner": [], "outer": []}
            for p in splits:
                cal, tst = p[:nc], p[nc:]
                ki = crc_index(Lin[cal], a, "nonincreasing"); ko = crc_index(Lout[cal], a, "nondecreasing")
                acc["inner"].append(ev.mean(tst, ki)); acc["outer"].append(ev.mean(tst, ko))
                risk["inner"].append(Lin[tst, ki].mean()); risk["outer"].append(Lout[tst, ko].mean())
            add_rows(rows, ds, a, "inner", np.mean(acc["inner"], 0), {"crc_spurious": np.mean(risk["inner"])})
            add_rows(rows, ds, a, "outer", np.mean(acc["outer"], 0), {"crc_missed": np.mean(risk["outer"])})
        print(f"   simulated level {lv}: {n} phantoms, {len(ev.cache)} masks scored")


def load_ktc(root):
    """Real KTC2023 phantoms: relative-score inputs, registered truth masks and names.
    Returns None if the prepared data or the registration file is missing."""
    base = os.path.join(root, "conformal_eit")
    kdir = os.path.join(base, "data", "ktc")
    reg = os.path.join(base, "results", "exp3_ktc_registration.mat")
    if not (os.path.isdir(kdir) and os.path.exists(reg)):
        return None
    R, C, dx = load_ops(os.path.join(base, "data", "conf_ops_ktc.mat"))
    best = sio.loadmat(reg)["best"].ravel(); ang, flip = float(best[1]), bool(best[2])
    uref = sio.loadmat(os.path.join(kdir, "ref.mat"))["Uelref"].ravel()
    Ys, Ts, names = [], [], []
    files = sorted(glob.glob(os.path.join(kdir, "data*.mat")), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[-1]))
    for f in files:
        num = re.findall(r"\d+", os.path.basename(f))[-1]
        tf = os.path.join(kdir, f"true{num}.mat")
        if not os.path.exists(tf):
            continue
        d = sio.loadmat(f); u = next(d[k] for k in d if k.lower().startswith("uel")).ravel()
        g = sio.loadmat(tf); G = next(v for k, v in g.items() if not k.startswith("__") and np.ndim(v) == 2)
        t = gt_to_pixels(np.asarray(G, float), C, ang, flip)
        if t.any():                                   # drops the water-only phantom
            Ys.append(u - uref); Ts.append(t); names.append(f"data{num}")
    X = R @ np.array(Ys).T                            # GREIT images; the voltage scale cancels in the relative score
    return {"X": X, "T": np.array(Ts).T, "names": names, "grid": Grid(C, dx), "angle": ang, "mirror": flip}


def run_real(root, rows):
    K = load_ktc(root)
    if K is None:
        print("   (real data: data/ktc or exp3_ktc_registration.mat not found, skipped)"); return
    T, grid = K["T"], K["grid"]
    S = rel_score(K["X"]); n = S.shape[1]
    Lin, Lout = seg_loss_curves(S, T); ev = Evaluator(S, T, grid)
    add_rows(rows, "ktc_real", np.nan, "greit25", ev.mean(range(n), I25),
             {"crc_spurious": Lin[:, I25].mean(), "crc_missed": Lout[:, I25].mean()})
    for a in [0.10, 0.20]:
        mi, mo, ri, ro = [], [], [], []
        for j in range(n):                            # leave-one-out, as in exp3_ktc_real.m
            c = np.array([i for i in range(n) if i != j])
            ki = crc_index(Lin[c], a, "nonincreasing"); ko = crc_index(Lout[c], a, "nondecreasing")
            mi.append(ev(j, ki)); mo.append(ev(j, ko)); ri.append(Lin[j, ki]); ro.append(Lout[j, ko])
        add_rows(rows, "ktc_real", a, "inner", np.nanmean(mi, 0), {"crc_spurious": np.mean(ri)})
        add_rows(rows, "ktc_real", a, "outer", np.nanmean(mo, 0), {"crc_missed": np.mean(ro)})
    print(f"   real KTC2023: {n} phantoms, registration angle {K['angle']:g} deg, mirror {int(K['mirror'])}")


# ---------- output ----------
LABEL = {"dice": "Dice", "iou": "IoU", "sensitivity": "Sensitivity", "specificity": "Specificity",
         "precision": "Precision", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
         "mcc": "MCC", "ssim": "SSIM", "boundary_f1": "Boundary F1 (1 pixel)",
         "hd95": "HD95 (tank radii)", "assd": "ASSD (tank radii)", "area_ratio": "Area / true area"}


def lookup(rows, ds, method, metric, alpha=None):
    for d, a, m, k, v in rows:
        if d == ds and m == method and k == metric and (alpha is None or abs(a - alpha) < 1e-9):
            return v
    return np.nan


def write_outputs(rows, res_dir, alpha):
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "exp5_seg_metrics.csv"), "w") as f:
        f.write("dataset,alpha,method,metric,value\n")
        for d, a, m, k, v in rows:
            f.write(f"{d},{'' if np.isnan(a) else a},{m},{k},{v:.6f}\n")
    datasets = [d for d in ["sim_level0", "sim_level1", "sim_level2", "sim_level3", "ktc_real"]
                if any(r[0] == d for r in rows)]
    lines = []
    for ds in datasets:
        print(f"\n{ds}  (alpha = {alpha:.2f})")
        print(f"{'metric':24s}{'GREIT 25%':>11s}{'inner set':>11s}{'outer set':>11s}")
        for k in METRICS:
            g, i, o = lookup(rows, ds, "greit25", k), lookup(rows, ds, "inner", k, alpha), lookup(rows, ds, "outer", k, alpha)
            print(f"{LABEL[k]:24s}{g:11.3f}{i:11.3f}{o:11.3f}")
            lines.append((ds, LABEL[k], g, i, o))
    with open(os.path.join(res_dir, "exp5_seg_metrics_table.tex"), "w") as f:
        for ds in datasets:
            f.write(f"% {ds}, alpha = {alpha:.2f}\n\\begin{{tabular}}{{lccc}}\n\\toprule\n"
                    " & GREIT & \\multicolumn{2}{c}{Conformal GREIT}\\\\\n\\cmidrule(lr){3-4}\n"
                    "Metric & 25\\% rule & Inner set & Outer set\\\\\n\\midrule\n")
            for d, lab, g, i, o in lines:
                if d == ds:
                    f.write(f"{lab} & {g:.3f} & {i:.3f} & {o:.3f}\\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap.add_argument("--alpha", type=float, default=0.10, help="alpha shown in the printed and LaTeX tables")
    ap.add_argument("--out", default=None, help="output folder (default: <root>/conformal_eit/results)")
    args = ap.parse_args()
    rows = []
    print("> Experiment 5: segmentation metrics")
    run_simulated(args.root, rows)
    run_real(args.root, rows)
    write_outputs(rows, args.out or os.path.join(args.root, "conformal_eit", "results"), args.alpha)
