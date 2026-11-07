function T2 = exp2_intervals(DATA, bay, cfg, res_dir, bae)
% EXP2_INTERVALS  Pixel error bars for the conductivity change under increasing
% model error, for three point estimates xhat (all in conductivity units):
%   Bayes : linear-Gaussian posterior mean          (field Xb)
%   GREIT : GREIT image times one scalar fitted on the tuning set (field Xgs)
%   TV    : total-variation reconstruction           (field Xt)
% Methods:
%   Bayes credible     xb +- z * s_post                       (Bayes only)
%   CRC all-pixels     xhat +- lam                            (Bayes only; shows the pitfall)
%   CRC grouped        xhat +- lam_g      (separate lam for active / background pixels)
%   CRC grp adaptive   xhat +- lam_g * (c0 + c1|xhat|), c fitted on half the calibration set
% Active pixels: |xhat| >= 25% of max|xhat| (known at test time).
% Guarantees: grouped rows -> miscov_active <= alpha and miscov_background <= alpha.
% True-anomaly coverage is reported, not guaranteed.
est = {'Bayes', 'Xb'; 'BAE', 'Xbae'; 'GREIT', 'Xgs'; 'TV', 'Xt'};
if nargin < 5, bae = []; end
rows = {};
s = bay.s_post;
for L = 1:numel(DATA)
    D = DATA(L);
    [cal, tst] = split_indices(D.n, cfg.cal_frac, cfg.n_splits, cfg.seed + 100 + L);
    for e = 1:size(est, 1)
        if ~isfield(D, est{e, 2}) || isempty(D.(est{e, 2})), continue; end
        Xh = D.(est{e, 2});
        E = abs(D.XT - Xh);
        A = abs(Xh) >= 0.25 * max(abs(Xh), [], 1);
        one = ones(size(Xh));
        for a = cfg.alpha
            if e <= 2                     % model-based credible intervals
                if e == 1, sp = s; else, sp = bae.s_post; end
                z = sqrt(2) * erfinv(1 - a);
                m = metrics(E, D.T, A, sp * ones(1, D.n), z, z, 1:D.n);
                rows(end + 1, :) = [{D.level, est{e, 1}, [est{e, 1} ' credible'], a}, num2cell(m)]; %#ok<AGROW>
            end
            acc = zeros(cfg.n_splits, 6, 3);
            for k = 1:cfg.n_splits
                c = cal(k, :); t = tst(k, :);
                if e == 1
                    l = crc_interval_lambda(E(:, c), true(size(E(:, c))), a);
                    acc(k, :, 1) = metrics(E, D.T, A, one, l, l, t);
                end
                [la, lb] = grouped(E(:, c), A(:, c), a);
                acc(k, :, 2) = metrics(E, D.T, A, one, la, lb, t);
                h = floor(numel(c) / 2); cf = c(1:h); cc = c(h + 1:end);
                U = adaptive_scale(E(:, cf), Xh(:, cf), A(:, cf), Xh);
                [la, lb] = grouped(E(:, cc) ./ U(:, cc), A(:, cc), a);
                acc(k, :, 3) = metrics(E, D.T, A, U, la, lb, t);
            end
            names = {'CRC all-pixels', 'CRC grouped', 'CRC grp adaptive'};
            for q = 1:3
                if q == 1 && e ~= 1, continue; end
                rows(end + 1, :) = [{D.level, est{e, 1}, names{q}, a}, num2cell(mean(acc(:, :, q), 1))]; %#ok<AGROW>
            end
        end
    end
end
T2 = cell2table(rows, 'VariableNames', {'level', 'estimate', 'method', 'alpha', 'miscov_all', ...
    'miscov_active', 'miscov_background', 'miscov_true_anomaly', 'width_active', 'width_true_anomaly'});
fprintf('\n===== Experiment 2: pixel error bars under model error =====\n');
fprintf('Guarantee for CRC grouped rows: miscov_active and miscov_background <= alpha.\n');
fprintf('Widths in conductivity units (smaller = more informative). True-anomaly columns are not guaranteed.\n');
disp(T2);
writetable(T2, fullfile(res_dir, 'exp2_intervals.csv'));

% ---- figure: Bayes credible vs adaptive CRC for each estimate, alpha = 0.10 ----
a0 = cfg.alpha(min(2, numel(cfg.alpha)));
f = figure('Color', 'w', 'Position', [100 100 1350 380]);
sel_rows = {{'Bayes', 'Bayes credible'}, {'BAE', 'BAE credible'}, {'Bayes', 'CRC grp adaptive'}, ...
    {'BAE', 'CRC grp adaptive'}, {'GREIT', 'CRC grp adaptive'}, {'TV', 'CRC grp adaptive'}};
cols = {'miscov_active', 'width_active', 'miscov_true_anomaly'};
ylab = {'miscoverage, active pixels', 'interval width, active pixels', 'miscoverage, true anomaly (not guaranteed)'};
for p = 1:3
    subplot(1, 3, p); hold on; grid on;
    for r = 1:numel(sel_rows)
        sel = strcmp(T2.estimate, sel_rows{r}{1}) & strcmp(T2.method, sel_rows{r}{2}) ...
            & abs(T2.alpha - a0) < 1e-9;
        if ~any(sel), continue; end
        plot(T2.level(sel), T2.(cols{p})(sel), '-o', 'LineWidth', 1.4, ...
            'DisplayName', [sel_rows{r}{1} ' ' sel_rows{r}{2}]);
    end
    if p == 1, plot(xlim, [a0 a0], 'k--', 'DisplayName', 'target \alpha'); end
    xlabel('model-error level'); xticks(unique(T2.level)); ylabel(ylab{p});
    title(sprintf('\\alpha = %.2f', a0)); legend('Location', 'best', 'FontSize', 7);
end
saveas(f, fullfile(res_dir, 'exp2_model_error.png'));
end

function [la, lb] = grouped(Rn, A, a)
la = crc_interval_lambda(Rn, A, a);
lb = crc_interval_lambda(Rn, ~A, a);
end

function U = adaptive_scale(Ef, Xf, Af, Xall)
% |error| ~ c0 + c1*|xhat| fitted by least squares on active pixels of the fit set
e = Ef(Af); x = abs(Xf(Af));
if numel(e) < 3, U = ones(size(Xall)); return; end
c = [ones(numel(x), 1), x] \ e;
c(1) = max(c(1), 1e-3 * mean(e)); c(2) = max(c(2), 0);
U = c(1) + c(2) * abs(Xall);
end

function m = metrics(E, T, A, SC, la, lb, idx)
% m = [miscov_all, miscov_active, miscov_background, miscov_true, width_active, width_true]
v = zeros(numel(idx), 6); ok = true(numel(idx), 6);
for j = 1:numel(idx)
    i = idx(j);
    lam = lb * ones(size(E, 1), 1); lam(A(:, i)) = la;
    hw = lam .* SC(:, i);
    miss = E(:, i) > hw;
    a = A(:, i); t = T(:, i);
    v(j, 1) = mean(miss);
    if any(a), v(j, 2) = mean(miss(a)); v(j, 5) = mean(2 * hw(a)); else, ok(j, [2 5]) = false; end
    if any(~a), v(j, 3) = mean(miss(~a)); else, ok(j, 3) = false; end
    if any(t), v(j, 4) = mean(miss(t)); v(j, 6) = mean(2 * hw(t)); else, ok(j, [4 6]) = false; end
end
m = zeros(1, 6);
for q = 1:6, m(q) = mean(v(ok(:, q), q)); end
end
