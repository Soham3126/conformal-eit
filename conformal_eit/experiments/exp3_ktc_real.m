function T3 = exp3_ktc_real(cfg, root, res_dir)
% EXP3_KTC_REAL  Claim 4: simulation-to-real transfer on the KTC2023 tank data.
%   (a) calibrate on SIMULATED phantoms of the KTC geometry, test on real tanks
%   (b) calibrate on REAL tanks, leave-one-out (honest small-sample estimate)
%   (c) fixed Graz 25% rule
% Reconstructor: GREIT (3D cylinder, one ring of 32 electrodes, KTC patterns).
K = ktc_read(fullfile(root, cfg.ktc.dir));
[fm_rec, fm_sim] = ktc_models(K, cfg);
ops = ops_build_generic(fm_rec, fullfile(root, 'data', 'conf_ops_ktc.mat'));
C = ops.C;

% ---- map real voltages into model units (one scalar fitted on the reference) ----
vh = fwd_solve(mk_image(fm_rec, 1)); vh = vh.meas;
scl = (vh' * K.Uelref(:)) / (vh' * vh);
Y = (K.Uel - K.Uelref(:)) / scl;                      % M x n, model units
n = size(Y, 2);
G = ops.R * Y;

% ---- noise level, estimated from the real data themselves ----
% The part of each difference vector outside the range of the Jacobian cannot
% come from any conductivity change, so its rms estimates the noise (plus
% model error, which makes the estimate conservative). Median over phantoms.
water = cellfun(@(g) ~any(g(:) > 0), K.gt);
[U, Sv, ~] = svd(ops.Jp, 'econ'); sv = diag(Sv);
r = sum(sv > sv(1) * 1e-8); U = U(:, 1:r);
res = Y - U * (U' * Y);
sig_each = sqrt(sum(res.^2, 1) / max(size(Y, 1) - r, 1));
sigma_m = median(sig_each(~water));
if any(water) && std(Y(:, find(water, 1))) > 0
    fprintf('> Water-phantom std (for comparison): %.3g\n', std(Y(:, find(water, 1))));
end
fprintf('> Noise std from out-of-range residuals: %.3g (model units, rank %d of %d)\n', ...
    sigma_m, r, size(Y, 1));
if ~(sigma_m > 0)
    sigma_m = 1e-3 * sqrt(mean((K.Uelref(:) / scl).^2));
    warning('%s', 'Noise estimate failed; noise std set to 0.1% of reference rms.');
end
ev = find(~water);

% ---- ground-truth registration: rotation (11.25 deg steps) x mirror ----
Srel = seg_scores(G, 'rel');
best = [-1 0 0];
for fl = 0:1
    for ang = 0:11.25:348.75
        dsc = zeros(numel(ev), 1);
        for j = 1:numel(ev)
            t = gt_to_pixels(K.gt{ev(j)}, C, ang, fl); s = Srel(:, ev(j)) >= 0.25;
            dsc(j) = 2 * sum(t & s) / max(sum(t) + sum(s), 1);
        end
        if mean(dsc) > best(1), best = [mean(dsc), ang, fl]; end
    end
end
fprintf('> GT registration: rotation %.2f deg, mirror %d, mean Dice of Graz 25%% set = %.2f\n', ...
    best(2), best(3), best(1));
if best(1) < 0.3
    warning('Low registration Dice: check that ground truths match the data files.');
end
T = false(ops.P, numel(ev));
for j = 1:numel(ev), T(:, j) = gt_to_pixels(K.gt{ev(j)}, C, best(2), best(3)); end
keep = any(T, 1); ev = ev(keep); T = T(:, keep); nr = numel(ev);
fprintf('> %d real phantoms with inclusions used for evaluation\n', nr);

% ---- simulated calibration data in the KTC geometry ----
cs = cfg; cs.contrast_pos = cfg.ktc.contrast_pos; cs.contrast_neg = cfg.ktc.contrast_neg;
cs.cylinder = true;                        % KTC targets run through the water column
sim = finish_sim2d(fm_sim);
D = gen_dataset(sim, ops, cs, cfg.ktc.N_sim, cfg.seed + 500, sigma_m, ...
    fullfile(root, 'data', 'conf_ktc_sim.mat'));
Gs = ops.R * D.Y;

rows = {};
for q = {'rel', 'snr'}
    kind = q{1};
    Ss = seg_scores(Gs, kind, sigma_m * ops.sig_px);
    Sr = seg_scores(G(:, ev), kind, sigma_m * ops.sig_px);
    if strcmp(kind, 'rel'), lgrid = linspace(0, 1.0001, cfg.grid_K);
    else, lgrid = linspace(0, 1.0001 * max([Ss(:); Sr(:)]), cfg.grid_K); end
    [Lin_s, Lout_s] = seg_loss_curves(Ss, D.T, lgrid);
    [Lin_r, Lout_r, Ar] = seg_loss_curves(Sr, T, lgrid);
    if strcmp(kind, 'rel')
        [~, i25] = min(abs(lgrid - 0.25));
        rows(end + 1, :) = {'Graz 25%', kind, NaN, mean(Lin_r(:, i25)), mean(Lout_r(:, i25)), ...
            mean(Ar(:, i25)), NaN}; %#ok<AGROW>
    end
    for a = cfg.ktc.alpha
        [~, ii] = crc_lambda(Lin_s, lgrid, a, 1, 'nonincreasing');
        [~, io] = crc_lambda(Lout_s, lgrid, a, 1, 'nondecreasing');
        rows(end + 1, :) = {'CRC sim-calibrated', kind, a, mean(Lin_r(:, ii)), ...
            mean(Lout_r(:, io)), mean(Ar(:, ii)), mean(Ar(:, io))}; %#ok<AGROW>
        ri = zeros(nr, 1); ro = ri; ai = ri; ao = ri;
        for j = 1:nr
            c = setdiff(1:nr, j);
            [~, ii] = crc_lambda(Lin_r(c, :), lgrid, a, 1, 'nonincreasing');
            [~, io] = crc_lambda(Lout_r(c, :), lgrid, a, 1, 'nondecreasing');
            ri(j) = Lin_r(j, ii); ro(j) = Lout_r(j, io); ai(j) = Ar(j, ii); ao(j) = Ar(j, io);
        end
        rows(end + 1, :) = {'CRC real LOO', kind, a, mean(ri), mean(ro), mean(ai), mean(ao)}; %#ok<AGROW>
    end
end
T3 = cell2table(rows, 'VariableNames', {'method', 'score', 'alpha', ...
    'spurious_inner', 'missed_outer', 'area_inner', 'area_outer'});
fprintf('\n===== Experiment 3: real tank data (KTC2023) =====\n');
fprintf('Real-data sample is small (n = %d): treat realised risks as estimates with wide error bars.\n', nr);
disp(T3);
writetable(T3, fullfile(res_dir, 'exp3_ktc_real.csv'));
save(fullfile(res_dir, 'exp3_ktc_registration.mat'), 'best', 'ev');
end

function sim = finish_sim2d(fm)
sim.fmdl = fm; sim.desc = 'KTC geometry, fine mesh';
sim.img_bg = mk_image(fm, 1.0);
v = fwd_solve(sim.img_bg); sim.vh = v.meas;
n = fm.nodes; e = fm.elems;
sim.ec = zeros(size(e, 1), size(n, 2));
for k = 1:size(e, 2), sim.ec = sim.ec + n(e(:, k), :); end
sim.ec = sim.ec / size(e, 2);
end
