% main_conformal.m
% =========================================================================
%  Conformal anomaly segmentation and error bars for EIT  -  full pipeline
% =========================================================================
%  Exp 1  guaranteed inner/outer anomaly sets vs the Graz 25% rule
%  Exp 2  pixel error bars: conformal vs Bayesian under 4 levels of model error
%  Exp 3  simulation-to-real transfer on KTC2023 tank data (if present)
%  Exp 4  region-level intervals (total change, anomaly area)
%
%  From the repository root:   run('conformal_eit/main_conformal.m')
%  QUICK = true  -> about 15-25 min smoke test
%  QUICK = false -> full run, roughly 2-3 h the first time, minutes once cached
% =========================================================================
clear; close all;
QUICK   = false;     % <-- true = 15-25 min smoke test
RUN_KTC = true;      % runs Exp 3 only if conformal_eit/data/ktc exists

here = fileparts(mfilename('fullpath'));
proj = fileparts(here);
cd(proj);
run(fullfile(proj, 'init_env.m'));
addpath(fullfile(here, 'conformal'), fullfile(here, 'recon'), fullfile(here, 'sim'), ...
    fullfile(here, 'real'), fullfile(here, 'experiments'));
cfg = cfg_default(QUICK);
tag = ''; if QUICK, tag = '_quick'; end
dat_dir = fullfile(here, 'data'); res_dir = fullfile(here, ['results' tag]);
if ~exist(dat_dir, 'dir'), mkdir(dat_dir); end
if ~exist(res_dir, 'dir'), mkdir(res_dir); end
t_all = tic;

%% 1. reconstruction operators (3D cylinder, 16 electrodes)
imdl = mk_common_model('b3cr', [16 1]);
ops = ops_build_generic(imdl.fwd_model, fullfile(dat_dir, 'conf_ops_b3cr.mat'));

%% 2. noise level and Bayesian baseline (tuned on its own model, level 0)
sim0 = make_sim_3d(0, ops, cfg);
sim1 = make_sim_3d(1, ops, cfg);
sigma_m = ref_sigma(sim1, ops, cfg.snr_db);
Dt = gen_dataset(sim0, ops, cfg, cfg.N_tune, cfg.seed + 900, sigma_m, ...
    fullfile(dat_dir, ['conf_tune' tag '.mat']));
bay = bayes_tune(ops, sigma_m, Dt.Y, Dt.XT, cfg.tau_grid);
Db = gen_dataset(sim1, ops, cfg, cfg.N_bae, cfg.seed + 700, sigma_m, ...
    fullfile(dat_dir, ['conf_bae' tag '.mat']));
hb = floor(Db.n / 2);
bae = bae_build(ops, sigma_m, Db.Y(:, 1:hb), Db.XT(:, 1:hb), cfg.tau_grid, ...
    Db.Y(:, hb + 1:end), Db.XT(:, hb + 1:end));
Gt = ops.R * Dt.Y;
a_greit = (Gt(:)' * Dt.XT(:)) / (Gt(:)' * Gt(:));      % GREIT -> conductivity units
fprintf('> GREIT amplitude scale fitted on the tuning set: %.4g\n', a_greit);
tvf = fullfile(dat_dir, ['conf_tv_tune' tag '.mat']);
if exist(tvf, 'file'), S = load(tvf); mu_tv = S.mu_tv; fprintf('> TV weight mu = %g (cached)\n', mu_tv);
else, mu_tv = tv_tune(ops, sigma_m, Dt.Y, Dt.XT, cfg.tv_mu_grid); save(tvf, 'mu_tv'); end

%% 3. datasets at each model-error level
DATA = struct([]);
for k = 1:numel(cfg.levels)
    lv = cfg.levels(k);
    if lv == 0, sm = sim0; elseif lv == 1, sm = sim1; else, sm = make_sim_3d(lv, ops, cfg); end
    fprintf('> Level %d: %s\n', lv, sm.desc);
    D = gen_dataset(sm, ops, cfg, cfg.N, cfg.seed + 10 * lv, sigma_m, ...
        fullfile(dat_dir, sprintf('conf_level%d%s.mat', lv, tag)));
    D.Xg = ops.R * D.Y;          % GREIT images
    D.Xb = bay.G * D.Y;          % Bayesian posterior means
    D.Xbae = bae.G * (D.Y - bae.mu_e);   % BAE posterior means
    D.Xgs = a_greit * D.Xg;      % GREIT in conductivity units
    tf = fullfile(dat_dir, sprintf('conf_tv_level%d%s.mat', lv, tag));
    S = []; if exist(tf, 'file'), S = load(tf); end
    if ~isempty(S) && S.mu_tv == mu_tv && size(S.Xt, 2) == D.n
        D.Xt = S.Xt;
    else
        fprintf('> TV reconstructions, level %d (%d phantoms)...\n', lv, D.n); t0 = tic;
        Xt = tv_recon(D.Y, ops, sigma_m, mu_tv); save(tf, 'Xt', 'mu_tv'); D.Xt = Xt;
        fprintf('   done in %.0fs\n', toc(t0));
    end
    D.level = lv;
    D = rmfield(D, 'Yc');
    if isempty(DATA), DATA = D; else, DATA(k) = D; end
end

%% 4. experiments
T1 = exp1_segmentation(DATA, ops, bay, cfg, res_dir);
T2 = exp2_intervals(DATA, bay, cfg, res_dir, bae);
T4 = exp4_region(DATA, ops, bay, bae, cfg, res_dir);
plot_examples(DATA(min(2, numel(DATA))), ops, cfg, res_dir);
T3 = [];
ktc_raw = fullfile(here, 'data', 'ktc_download');
if RUN_KTC && ~exist(fullfile(here, cfg.ktc.dir, 'ref.mat'), 'file') && exist(ktc_raw, 'dir')
    prepare_ktc(ktc_raw, fullfile(here, cfg.ktc.dir));   % first run after download
end
if RUN_KTC && exist(fullfile(here, cfg.ktc.dir), 'dir')
    try
        T3 = exp3_ktc_real(cfg, here, res_dir);
    catch ME
        fprintf(2, '\nExperiment 3 stopped: %s\n(Experiments 1-2 are saved.)\n', ME.message);
    end
else
    fprintf('\n(Experiment 3 skipped: put the KTC2023 files in %s)\n', fullfile(here, cfg.ktc.dir));
end

save(fullfile(res_dir, 'all_results.mat'), 'T1', 'T2', 'T3', 'T4', 'cfg', 'sigma_m');
fprintf('\n> Done in %.1f min. Tables (CSV) and figures (PNG) are in %s\n', toc(t_all) / 60, res_dir);
