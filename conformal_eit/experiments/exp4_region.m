function T4 = exp4_region(DATA, ops, bay, bae, cfg, res_dir)
% EXP4_REGION  Guaranteed intervals for two scalar, clinically meaningful
% quantities, under increasing model error:
%   total  : sum of the conductivity change over the image (the EIT analogue of
%            a lung "tidal" signal), in conductivity x pixel units. Evaluated only
%            on phantoms whose inclusions all share one sign (as in lung EIT);
%            with mixed signs the total cancels towards zero and is meaningless.
%   area   : true anomaly area (pixels with x ~= 0) as a fraction of the tank
% Methods:
%   Bayes / BAE credible : qhat_B +- z * sqrt(1' Sigma 1)       (total only)
%   CRC (GREIT, Bayes, BAE, TV): split conformal with relative scale
%       q_cal = a * qhat (a fitted on half of the calibration set),
%       interval q_cal +- lam * max(|q_cal|, floor)
% Guarantee for CRC rows: miscoverage <= alpha.  rel_width = width / |q_true|.
est = {'GREIT', 'Xgs'; 'Bayes', 'Xb'; 'BAE', 'Xbae'; 'TV', 'Xt'};
rows = {};
for L = 1:numel(DATA)
    D = DATA(L);
    same = cellfun(@(t) all(sign(t(:, 4)) == sign(t(1, 4))), D.tg);
    qt.total = sum(D.XT, 1);
    qt.area = mean(D.T, 1);
    for qn = {'total', 'area'}
        if strcmp(qn{1}, 'total'), sel = find(same); else, sel = 1:D.n; end
        [cal, tst] = split_indices(numel(sel), cfg.cal_frac, cfg.n_splits, cfg.seed + 300 + L);
        cal = sel(cal); tst = sel(tst);
        if strcmp(qn{1}, 'total')
            fprintf('   level %d: %d of %d phantoms have single-sign inclusions\n', D.level, numel(sel), D.n);
        end
        q = qt.(qn{1});
        for a = cfg.alpha
            z = sqrt(2) * erfinv(1 - a);
            if strcmp(qn{1}, 'total')     % model-based credible intervals
                for b = {'Bayes', 'BAE'}
                    if strcmp(b{1}, 'Bayes'), mdl = bay; X = D.Xb; else, mdl = bae; X = D.Xbae; end
                    if isempty(mdl) || isempty(X), continue; end
                    if isfield(mdl, 'Sigma'), sd = sqrt(sum(mdl.Sigma(:)));
                    else, sd = NaN; end
                    if isnan(sd), continue; end
                    qh = sum(X, 1); hw = z * sd;
                    rows(end + 1, :) = {D.level, qn{1}, b{1}, [b{1} ' credible'], a, ...
                        mean(abs(q(sel) - qh(sel)) > hw), median(2 * hw ./ max(abs(q(sel)), eps))}; %#ok<AGROW>
                end
            end
            for e = 1:size(est, 1)
                if ~isfield(D, est{e, 2}) || isempty(D.(est{e, 2})), continue; end
                X = D.(est{e, 2});
                if strcmp(qn{1}, 'total'), qh = sum(X, 1);
                else, qh = mean(abs(X) >= 0.25 * max(abs(X), [], 1), 1); end   % 25%-set area
                mis = zeros(cfg.n_splits, 1); rw = mis;
                for k = 1:cfg.n_splits
                    c = cal(k, :); t = tst(k, :);
                    h = floor(numel(c) / 2); cf = c(1:h); cc = c(h + 1:end);
                    sl = (qh(cf) * q(cf)') / max(qh(cf) * qh(cf)', eps);   % linear recalibration
                    fl = 0.05 * median(abs(q(cf)));                       % scale floor
                    sc = @(ix) max(abs(sl * qh(ix)), fl);
                    lam = crc_interval_lambda(abs(q(cc) - sl * qh(cc)) ./ sc(cc), true(1, numel(cc)), a);
                    hwt = lam * sc(t);
                    mis(k) = mean(abs(q(t) - sl * qh(t)) > hwt);
                    rw(k) = median(2 * hwt ./ max(abs(q(t)), eps));
                end
                rows(end + 1, :) = {D.level, qn{1}, est{e, 1}, 'CRC', a, mean(mis), mean(rw)}; %#ok<AGROW>
            end
        end
    end
end
T4 = cell2table(rows, 'VariableNames', {'level', 'quantity', 'estimate', 'method', 'alpha', ...
    'miscoverage', 'median_rel_width'});
fprintf('\n===== Experiment 4: region-level intervals (total change, anomaly area) =====\n');
fprintf('CRC rows: miscoverage <= alpha guaranteed. median_rel_width = interval width / |true value|.\n');
disp(T4);
writetable(T4, fullfile(res_dir, 'exp4_region.csv'));

% ---- figure: total change, alpha = 0.10 ----
a0 = cfg.alpha(min(2, numel(cfg.alpha)));
f = figure('Color', 'w', 'Position', [100 100 900 380]);
sub = T4(strcmp(T4.quantity, 'total') & abs(T4.alpha - a0) < 1e-9, :);
tags = unique(strcat(sub.estimate, {' '}, sub.method), 'stable');
for p = 1:2
    subplot(1, 2, p); hold on; grid on;
    for t = 1:numel(tags)
        m = strcmp(strcat(sub.estimate, {' '}, sub.method), tags{t});
        if p == 1, v = sub.miscoverage(m); else, v = sub.median_rel_width(m); end
        plot(sub.level(m), v, '-o', 'LineWidth', 1.4, 'DisplayName', tags{t});
    end
    xlabel('model-error level'); xticks(unique(sub.level));
    if p == 1, plot(xlim, [a0 a0], 'k--', 'DisplayName', 'target'); ylabel('miscoverage');
    else, ylabel('median interval width / |true total|'); end
    title(sprintf('Total conductivity change, \\alpha = %.2f', a0)); legend('Location', 'best', 'FontSize', 7);
end
saveas(f, fullfile(res_dir, 'exp4_region.png'));
end
