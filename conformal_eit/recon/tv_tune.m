function [mu_best, err] = tv_tune(ops, sigma_m, Y, XT, mu_grid)
% TV_TUNE  Pick the TV weight with the lowest mean relative error on a tuning set
% (same rule as bayes_tune, so both point estimates are tuned the same way).
err = zeros(size(mu_grid));
for k = 1:numel(mu_grid)
    X = tv_recon(Y, ops, sigma_m, mu_grid(k));
    err(k) = mean(sqrt(sum((X - XT).^2, 1)) ./ max(sqrt(sum(XT.^2, 1)), eps));
    fprintf('   TV mu = %-8g relative error %.3f\n', mu_grid(k), err(k));
end
[~, kb] = min(err); mu_best = mu_grid(kb);
if kb == 1 || kb == numel(mu_grid)
    warning('tv_tune: best mu at the edge of the grid (%g); widen cfg.tv_mu_grid.', mu_best);
end
fprintf('> TV weight mu = %g (tuning error %.3f)\n', mu_best, err(kb));
end
