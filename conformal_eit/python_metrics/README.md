# Python: segmentation metrics and figures

Uses the data already cached by `main_conformal.m` (and `run_exp3.m` for the real
phantoms); nothing is re-simulated. Run the MATLAB code first.

## Setup
```
cd conformal_eit/python_metrics
pip install -r requirements.txt
```

## Scripts
| Script | What it does | Output (in `conformal_eit/results/`) |
|---|---|---|
| `run_seg_metrics.py` | Scores the GREIT 25% rule and the conformal inner / outer sets with Dice, IoU, sensitivity, specificity, precision, accuracy, balanced accuracy, MCC, SSIM, boundary F1, HD95, ASSD and area ratio (about 2 min) | `exp5_seg_metrics.csv`, `exp5_seg_metrics_table.tex` |
| `make_fig_model_error.py` | Pixel error bars versus model-error level, from `exp2_intervals.csv` | `exp2_model_error.png` |
| `make_fig_ktc.py` | Real KTC2023 phantoms: truth, GREIT image, conformal sets (leave-one-out) | `fig_ktc_examples.png` |
| `seg_metrics.py` | Metric definitions used by `run_seg_metrics.py` | |

All scripts accept `--root C:/path/to/EIT_Hybrid_Project` if the folder is moved, and `--alpha`.

## Photograph row in the real-phantom figure
`make_fig_ktc.py` adds a top row of photographs if it finds
`ktc_photo_data1.jpg`, `ktc_photo_data17.jpg`, `ktc_photo_data24.jpg` and `ktc_photo_data9.jpg`
in `paper/figures/ktc_photos/` (or the folder given with `--photos`). It then writes the
figure to `paper/figures/fig_ktc_examples.png` instead. The photographs come from the
KTC2023 paper (Räsänen et al., Applied Mathematics for Modern Challenges 2(2), 2024), are
not part of the open dataset and are not distributed with this code.

## Protocol
- Simulated levels 0-3: 200 random 50/50 calibration/test splits, metrics averaged over the
  test phantoms and then over splits (same design as Experiment 1; the splits are drawn in
  Python, so they are not the identical splits MATLAB drew).
- Real KTC2023: leave-one-out over the 25 phantoms, same registration as Experiment 3.
- Distances (HD95, ASSD) are in units of the tank radius; one pixel is 1/16 of the radius.
