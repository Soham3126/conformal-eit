function [bay, tau_best, err] = bayes_tune(ops, sigma_m, Y, XT, tau_grid)
% BAYES_TUNE  Pick the prior scale tau that minimises the mean relative error
% of the posterior mean on a tuning set (gives the Bayesian baseline its best
% possible point estimate before its error bars are judged).
if nargin < 5 || isempty(tau_grid), tau_grid = logspace(-2, 1, 13); end
err = zeros(size(tau_grid));
for k = 1:numel(tau_grid)
    b = bayes_build(ops, sigma_m, tau_grid(k));
    X = b.G * Y;
    err(k) = mean(sqrt(sum((X - XT).^2, 1)) ./ max(sqrt(sum(XT.^2, 1)), eps));
end
[~, kb] = min(err);
tau_best = tau_grid(kb);
if kb == 1 || kb == numel(tau_grid)
    warning('bayes_tune: best tau at the edge of the grid (%g); widen tau_grid.', tau_best);
end
bay = bayes_build(ops, sigma_m, tau_best);
fprintf('> Bayesian prior scale tau = %g (tuning error %.3f)\n', tau_best, err(kb));
end
