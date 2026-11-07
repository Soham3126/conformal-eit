function T1 = exp1_segmentation(DATA, ops, bay, cfg, res_dir)
% EXP1_SEGMENTATION  Claim 1: conformal "inner" and "outer" anomaly sets with
% guaranteed error rates, versus the fixed Graz 25%-of-peak rule.
%   inner set  {s >= lam_in}  : E[spurious area / true area] <= alpha
%   outer set  {s >= lam_out} : E[missed fraction of target] <= alpha
% Rows: level x reconstructor (GREIT, Bayes mean) x score (rel, snr) x alpha.
rows = {};
recs = {'GREIT', 'Bayes', 'TV'};
kinds = {'rel', 'snr'};
for L = 1:numel(DATA)
    D = DATA(L);
    [cal, tst] = split_indices(D.n, cfg.cal_frac, cfg.n_splits, cfg.seed + L);
    for r = 1:3
        if r == 1, X = D.Xg; sc = D.sigma_m * ops.sig_px;
        elseif r == 2, X = D.Xb; sc = bay.s_post;
        else, X = D.Xt; sc = []; end
        for q = 1:2
            if r == 3 && q == 2, continue; end   % no noise-scale model for TV
            S = seg_scores(X, kinds{q}, sc);
            if q == 1, lgrid = linspace(0, 1.0001, cfg.grid_K);
            else, lgrid = linspace(0, 1.0001 * max(S(:)), cfg.grid_K); end
            [Lin, Lout, Area] = seg_loss_curves(S, D.T, lgrid);
            if q == 1   % fixed Graz rule for reference (no calibration)
                [~, i25] = min(abs(lgrid - 0.25));
                rows(end + 1, :) = {D.level, recs{r}, 'Graz 25%', NaN, ...
                    mean(Lin(:, i25)), mean(Lout(:, i25)), mean(Area(:, i25)), NaN, NaN}; %#ok<AGROW>
            end
            for a = cfg.alpha
                rin = zeros(cfg.n_splits, 1); rout = rin; ain = rin; aout = rin; viol = rin;
                for s = 1:cfg.n_splits
                    [~, iin]  = crc_lambda(Lin(cal(s, :), :),  lgrid, a, 1, 'nonincreasing');
                    [~, iout] = crc_lambda(Lout(cal(s, :), :), lgrid, a, 1, 'nondecreasing');
                    rin(s)  = mean(Lin(tst(s, :), iin));
                    rout(s) = mean(Lout(tst(s, :), iout));
                    ain(s)  = mean(Area(tst(s, :), iin));
                    aout(s) = mean(Area(tst(s, :), iout));
                    viol(s) = rin(s) > a || rout(s) > a;
                end
                rows(end + 1, :) = {D.level, recs{r}, ['CRC ' kinds{q}], a, ...
                    mean(rin), mean(rout), mean(ain), mean(aout), mean(viol)}; %#ok<AGROW>
            end
        end
    end
end
T1 = cell2table(rows, 'VariableNames', {'level', 'recon', 'method', 'alpha', ...
    'spurious_inner', 'missed_outer', 'area_inner', 'area_outer', 'split_violation_rate'});
fprintf('\n===== Experiment 1: guaranteed anomaly segmentation =====\n');
fprintf('spurious_inner = spurious area / true area of the INNER set (target <= alpha)\n');
fprintf('missed_outer   = missed target fraction of the OUTER set (target <= alpha)\n');
fprintf('area_* = set area / true area (1 = perfect size); Graz row: both from the one 25%% set\n');
disp(T1);
writetable(T1, fullfile(res_dir, 'exp1_segmentation.csv'));

% ---- figure: realised risk vs alpha (validity) at the most realistic level ----
f = figure('Color', 'w', 'Position', [100 100 900 380]);
Lsel = max([DATA.level]);
sub = T1(T1.level == Lsel & ~isnan(T1.alpha), :);
tags = unique(strcat(sub.recon, {' '}, sub.method), 'stable');
for p = 1:2
    subplot(1, 2, p); hold on; grid on;
    plot([0 0.25], [0 0.25], 'k--', 'DisplayName', 'target');
    for t = 1:numel(tags)
        m = strcmp(strcat(sub.recon, {' '}, sub.method), tags{t});
        if p == 1, v = sub.spurious_inner(m); else, v = sub.missed_outer(m); end
        plot(sub.alpha(m), v, '-o', 'LineWidth', 1.4, 'DisplayName', tags{t});
    end
    xlabel('\alpha (target risk)');
    if p == 1, ylabel('realised spurious area (inner set)'); else, ylabel('realised missed fraction (outer set)'); end
    title(sprintf('Level %d', Lsel)); legend('Location', 'northwest');
end
saveas(f, fullfile(res_dir, 'exp1_validity.png'));
end
