"""Segmentation metrics for binary anomaly masks on the EIT pixel grid.

Every function takes boolean vectors over the P tank pixels (pred = pixels the
method marks as anomaly, true = ground-truth anomaly). Region and pixel-count
metrics use the tank pixels only. Boundary metrics and SSIM need the 2D layout,
which is supplied by a Grid object built from the pixel centres.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

METRICS = ["dice", "iou", "sensitivity", "specificity", "precision", "accuracy",
           "balanced_accuracy", "mcc", "ssim", "boundary_f1", "hd95", "assd", "area_ratio"]
HIGHER_IS_BETTER = {m: True for m in METRICS}
HIGHER_IS_BETTER.update({"hd95": False, "assd": False, "area_ratio": None})  # area_ratio: 1 is ideal


class Grid:
    """Maps a P-vector of tank pixels onto the n x n image used for plotting."""

    def __init__(self, centres, dx):
        self.dx = float(dx)
        self.n = int(round(2.0 / self.dx))
        # same rule as img32() in the MATLAB code (1-based there, 0-based here)
        self.ix = np.floor((centres[:, 0] + 1) / self.dx + 0.5 + 0.5).astype(int) - 1
        self.iy = np.floor((centres[:, 1] + 1) / self.dx + 0.5 + 0.5).astype(int) - 1
        self.tank = np.zeros((self.n, self.n), bool)
        self.tank[self.iy, self.ix] = True

    def image(self, v):
        img = np.zeros((self.n, self.n), dtype=np.asarray(v).dtype)
        img[self.iy, self.ix] = v
        return img


_CROSS = ndi.generate_binary_structure(2, 1)  # 4-neighbour


def _boundary(mask):
    """Pixels of the mask that touch a non-mask pixel (4-neighbour)."""
    return mask & ~ndi.binary_erosion(mask, _CROSS, border_value=0)


def _safe(num, den):
    return num / den if den > 0 else np.nan


def region_metrics(pred, true):
    pred = np.asarray(pred, bool); true = np.asarray(true, bool)
    tp = float(np.sum(pred & true)); fp = float(np.sum(pred & ~true))
    fn = float(np.sum(~pred & true)); tn = float(np.sum(~pred & ~true))
    sens = _safe(tp, tp + fn); spec = _safe(tn, tn + fp)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "dice": _safe(2 * tp, 2 * tp + fp + fn),
        "iou": _safe(tp, tp + fp + fn),
        "sensitivity": sens,
        "specificity": spec,
        "precision": _safe(tp, tp + fp),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (sens + spec),
        "mcc": _safe(tp * tn - fp * fn, den),
        "area_ratio": _safe(tp + fp, tp + fn),
    }


def boundary_metrics(pred_img, true_img, dx, tol_px=1.0):
    """Hausdorff-95, average symmetric surface distance (tank-radius units) and
    boundary F1 (share of boundary pixels within tol_px pixels of the other boundary)."""
    out = {"hd95": np.nan, "assd": np.nan, "boundary_f1": np.nan}
    if not pred_img.any() or not true_img.any():
        if true_img.any() or pred_img.any():
            out["boundary_f1"] = 0.0
        return out
    bp, bt = _boundary(pred_img), _boundary(true_img)
    d_to_t = ndi.distance_transform_edt(~bt)     # distance of every pixel to the true boundary
    d_to_p = ndi.distance_transform_edt(~bp)
    dp, dt = d_to_t[bp], d_to_p[bt]              # pred->true and true->pred boundary distances (pixels)
    out["hd95"] = float(np.percentile(np.hstack((dp, dt)), 95)) * dx
    out["assd"] = 0.5 * (dp.mean() + dt.mean()) * dx
    prec, rec = np.mean(dp <= tol_px), np.mean(dt <= tol_px)
    out["boundary_f1"] = _safe(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
    return out


def ssim_metric(pred_img, true_img, tank):
    """SSIM between the two binary images (Gaussian window, sigma 1.5), averaged
    over the tank pixels only."""
    _, smap = structural_similarity(true_img.astype(float), pred_img.astype(float), data_range=1.0,
                                    gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False, full=True)
    return float(smap[tank].mean())


def all_metrics(pred, true, grid):
    m = region_metrics(pred, true)
    pi, ti = grid.image(np.asarray(pred, bool)), grid.image(np.asarray(true, bool))
    m.update(boundary_metrics(pi, ti, grid.dx))
    m["ssim"] = ssim_metric(pi, ti, grid.tank)
    return np.array([m[k] for k in METRICS], float)
