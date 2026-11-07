% run_exp3.m  -  prepare the KTC2023 data and run Experiment 3 only.
% Expects the download in conformal_eit/data/ktc_download/ (Codes_Matlab/,
% EvaluationData_full/). Run:  run('conformal_eit/run_exp3.m')
clear; close all;
QUICK = false;
here = fileparts(mfilename('fullpath'));
proj = fileparts(here);
cd(proj);
run(fullfile(proj, 'init_env.m'));
addpath(fullfile(here, 'conformal'), fullfile(here, 'recon'), fullfile(here, 'sim'), ...
    fullfile(here, 'real'), fullfile(here, 'experiments'));
cfg = cfg_default(QUICK);
ktc = fullfile(here, cfg.ktc.dir);
if ~exist(fullfile(ktc, 'ref.mat'), 'file')
    prepare_ktc(fullfile(here, 'data', 'ktc_download'), ktc);
end
res_dir = fullfile(here, 'results'); if ~exist(res_dir, 'dir'), mkdir(res_dir); end
T3 = exp3_ktc_real(cfg, here, res_dir);
save(fullfile(res_dir, 'exp3_results.mat'), 'T3', 'cfg');
