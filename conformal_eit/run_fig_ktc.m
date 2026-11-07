% run_fig_ktc.m  -  Figure 3: real KTC2023 phantoms with ground truth, GREIT image
% and conformal inner / outer sets (alpha = 0.10, relative score). For each shown
% phantom, lambda is calibrated on the OTHER real phantoms (leave-one-out), exactly
% as in Experiment 3. Run after main_conformal.m:  run('conformal_eit/run_fig_ktc.m')
clear; close all;
here = fileparts(mfilename('fullpath')); proj = fileparts(here);
cd(proj); run(fullfile(proj, 'init_env.m'));
addpath(fullfile(here, 'conformal'), fullfile(here, 'recon'), fullfile(here, 'sim'), ...
    fullfile(here, 'real'), fullfile(here, 'experiments'));
cfg = cfg_default(false);
res_dir = fullfile(here, 'results');
alpha = 0.10;
n_show = 4;

K = ktc_read(fullfile(here, cfg.ktc.dir));
[fm_rec, ~] = ktc_models(K, cfg);
ops = ops_build_generic(fm_rec, fullfile(here, 'data', 'conf_ops_ktc.mat'));
vh = fwd_solve(mk_image(fm_rec, 1)); vh = vh.meas;
scl = (vh' * K.Uelref(:)) / (vh' * vh);
G = ops.R * ((K.Uel - K.Uelref(:)) / scl);

S = load(fullfile(res_dir, 'exp3_ktc_registration.mat'), 'best', 'ev');
ev = S.ev; ang = S.best(2); fl = S.best(3);
nr = numel(ev);
T = false(ops.P, nr);
for j = 1:nr, T(:, j) = gt_to_pixels(K.gt{ev(j)}, ops.C, ang, fl); end
Sr = seg_scores(G(:, ev), 'rel');
lgrid = linspace(0, 1.0001, cfg.grid_K);
[Lin, Lout] = seg_loss_curves(Sr, T, lgrid);

% ---- colours (pixels outside the tank are left white) ----
c_bg    = [0.88 0.88 0.88];   % tank, no anomaly
c_true  = [0.85 0.33 0.10];   % true anomaly
c_miss  = [0.75 0.10 0.10];   % true anomaly outside the outer set (missed)
c_outer = [0.60 0.78 0.95];   % in outer set only
c_inner = [0.08 0.30 0.65];   % in inner set
map_truth = [c_bg; c_true];
map_sets  = [c_miss; c_bg; c_outer; c_inner];   % values -1, 0, 1, 2

show = round(linspace(1, nr, n_show));            % spread across training + levels 1-7
f = figure('Color', 'w', 'Position', [50 50 230 * n_show 800]);
for k = 1:n_show
    j = show(k); c = setdiff(1:nr, j);
    li = crc_lambda(Lin(c, :), lgrid, alpha, 1, 'nonincreasing');
    lo = crc_lambda(Lout(c, :), lgrid, alpha, 1, 'nondecreasing');

    ax = subplot(3, n_show, k);
    show_img(ax, img32(double(T(:, j)), ops), [-0.5 1.5], map_truth);
    title(sprintf('truth (%s)', strrep(K.name{ev(j)}, '.mat', '')), ...
        'Interpreter', 'none', 'FontSize', 9);

    ax = subplot(3, n_show, n_show + k);
    show_img(ax, img32(abs(G(:, ev(j))), ops), [], parula(256));
    title('|GREIT|', 'FontSize', 9);

    m = double(Sr(:, j) >= lo) + double(Sr(:, j) >= li);   % 1 outer only, 2 inner
    m(T(:, j) & m == 0) = -1;                              % missed anomaly pixels
    ax = subplot(3, n_show, 2 * n_show + k);
    show_img(ax, img32(m, ops), [-1.5 2.5], map_sets);
    title(sprintf('\\lambda_{in}=%.2f, \\lambda_{out}=%.2f', li, lo), 'FontSize', 9);
end

% ---- legend for the bottom row ----
lab = {'true anomaly', 'inner set', 'outer set only', 'missed anomaly', 'tank (no anomaly)'};
col = {c_true, c_inner, c_outer, c_miss, c_bg};
hold(ax, 'on'); h = gobjects(1, numel(lab));
for i = 1:numel(lab)
    h(i) = patch(ax, NaN, NaN, col{i}, 'EdgeColor', [0.4 0.4 0.4]);
end
lg = legend(h, lab, 'Orientation', 'horizontal', 'FontSize', 9, 'Box', 'off');
lg.Units = 'normalized'; lg.Position = [0.05 0.01 0.9 0.04];

out = fullfile(res_dir, 'fig_ktc_examples.png');
try
    exportgraphics(f, out, 'Resolution', 300, 'BackgroundColor', 'white');
catch
    saveas(f, out);                                % MATLAB older than R2020a
end
fprintf('> Saved %s\n', out);

function show_img(ax, I, lims, map)
% Draws I with NaN (outside the tank) transparent on a white background.
im = imagesc(ax, I); set(im, 'AlphaData', ~isnan(I));
axis(ax, 'image', 'off'); set(ax, 'YDir', 'normal', 'Color', 'w');
colormap(ax, map);
if ~isempty(lims), clim_set(ax, lims); end
end

function clim_set(ax, lims)
try, clim(ax, lims); catch, caxis(ax, lims); end   %#ok<CAXIS> clim needs R2022a
end

function I = img32(x, ops)
n = round(2 / ops.dx); I = nan(n);
ix = round((ops.C(:, 1) + 1) / ops.dx + 0.5); iy = round((ops.C(:, 2) + 1) / ops.dx + 0.5);
I(sub2ind([n n], iy, ix)) = x;
end
