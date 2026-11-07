function cfg = cfg_default(quick)
% CFG_DEFAULT  All experiment settings in one place.
if nargin < 1, quick = false; end
cfg.quick       = quick;
cfg.seed        = 2026;
% ---- simulation (3D cylinder, 16 electrodes) ----
cfg.levels      = 0:3;           % model-error levels (see make_sim_3d)
cfg.N           = 400;           % phantoms per level
cfg.N_tune      = 60;            % phantoms for tuning the Bayesian prior (level 0)
cfg.N_bae       = 300;           % fine-mesh phantoms for the BAE baseline (half stats, half tuning)
cfg.snr_db      = 40;            % noise level on the reference target
cfg.n_incl_max  = 3;             % 1..3 inclusions per phantom
cfg.r_range     = [0.10 0.20];
cfg.max_rad     = 0.75;
cfg.p_negative  = 0.5;           % probability an inclusion is resistive
cfg.contrast_pos = [0.5 2.0];    % conductivity change of conductive inclusions
cfg.contrast_neg = [-0.7 -0.3];  % ... of resistive inclusions
cfg.jitter_deg  = 1.5;           % electrode position error sd (level >= 2)
cfg.radius_err  = 0.03;          % tank radius error (level 3)
% ---- conformal ----
cfg.alpha       = [0.05 0.10 0.20];
cfg.n_splits    = 200;           % random calibration/test splits
cfg.cal_frac    = 0.5;
cfg.grid_K      = 401;           % thresholds per score
cfg.tau_grid    = logspace(-2, 1, 13);
cfg.tv_mu_grid  = 2.^(-6:2:14);    % TV weight grid (tuned on the tuning set)
% ---- real data: Kuopio Tomography Challenge 2023 (2D, 32 electrodes) ----
cfg.ktc.dir         = fullfile('data', 'ktc');
cfg.ktc.recon_model = 'd2C';     % EIDORS 2D circle, complete electrode model
cfg.ktc.sim_model   = 'h2C';     % finer mesh for simulated calibration data
cfg.ktc.N_sim       = 400;
cfg.ktc.contrast_pos = [3 10];   % metal / conductive targets
cfg.ktc.contrast_neg = [-0.95 -0.8]; % plastic / resistive targets
cfg.ktc.alpha       = [0.10 0.20];
if quick
    cfg.levels = [0 1]; cfg.N = 80; cfg.N_tune = 20; cfg.n_splits = 50; cfg.N_bae = 80;
    cfg.ktc.N_sim = 80;
end
end
