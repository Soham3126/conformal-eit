function plot_examples(D, ops, cfg, res_dir, n_show)
% PLOT_EXAMPLES  Paper figure: truth, GREIT image, and the conformal inner /
% outer anomaly sets (alpha = 0.10, 'rel' score, calibrated on the other phantoms).
if nargin < 5, n_show = 4; end
a = 0.10;
S = seg_scores(D.Xg, 'rel');
lgrid = linspace(0, 1.0001, cfg.grid_K);
[Lin, Lout] = seg_loss_curves(S, D.T, lgrid);
show = 1:min(n_show, D.n);
cal = setdiff(1:D.n, show);
li = crc_lambda(Lin(cal, :), lgrid, a, 1, 'nonincreasing');
lo = crc_lambda(Lout(cal, :), lgrid, a, 1, 'nondecreasing');
f = figure('Color', 'w', 'Position', [50 50 260 * numel(show) 760]);
for k = 1:numel(show)
    i = show(k);
    subplot(3, numel(show), k); imagesc(img32(D.XT(:, i), ops)); axis image off;
    set(gca, 'YDir', 'normal'); title('truth');
    subplot(3, numel(show), numel(show) + k); imagesc(img32(D.Xg(:, i), ops)); axis image off;
    set(gca, 'YDir', 'normal'); title('GREIT');
    m = double(S(:, i) >= lo) + double(S(:, i) >= li);   % 1 = outer only, 2 = inner
    m(D.T(:, i) & m == 0) = -1;                          % missed target pixels
    subplot(3, numel(show), 2 * numel(show) + k); imagesc(img32(m, ops), [-1 2]); axis image off;
    set(gca, 'YDir', 'normal');
    title(sprintf('inner \\lambda=%.2f / outer \\lambda=%.2f', li, lo), 'FontSize', 8);
end
colormap(f, parula);
saveas(f, fullfile(res_dir, 'fig_examples.png'));
end

function I = img32(x, ops)
n = round(2 / ops.dx); I = nan(n);
ix = round((ops.C(:, 1) + 1) / ops.dx + 0.5); iy = round((ops.C(:, 2) + 1) / ops.dx + 0.5);
I(sub2ind([n n], iy, ix)) = x;
end
