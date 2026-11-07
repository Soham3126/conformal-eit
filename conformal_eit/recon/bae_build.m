function bae = bae_build(ops, sigma_m, Ytr, XTtr, tau_grid, Yval, XTval)
% BAE_BUILD  Bayesian approximation error (BAE) baseline (Kaipio & Somersalo).
% The forward-model error  e = y - Jp*x  (linearisation, discretisation and any
% other mismatch captured by the training simulations) is modelled as Gaussian,
% e ~ N(mu_e, Gamma_e), and added to the measurement noise:
%   y | x ~ N(Jp x + mu_e, sigma_m^2 I + Gamma_e)
% Posterior with prior x ~ N(0, tau^2 Q^-1), Q = Dg'Dg + eps I:
%   Sigma = (Jp' Gn^-1 Jp + Q/tau^2)^-1,  xb = Sigma Jp' Gn^-1 (y - mu_e)
% Ytr/XTtr: simulations from the best available model, used to estimate mu_e,
% Gamma_e. Yval/XTval: separate simulations from the same model to tune tau.
E = Ytr - ops.Jp * XTtr;                        % approximation errors (M x n)
bae.mu_e = mean(E, 2);
Ec = E - bae.mu_e;
Gam = (Ec * Ec') / max(size(E, 2) - 1, 1);
Gn = sigma_m^2 * eye(ops.M) + Gam;
Gn = (Gn + Gn') / 2;
Wn = Gn \ eye(ops.M);
Q = full(ops.Dg' * ops.Dg) + 1e-3 * eye(ops.P);
JtW = ops.Jp' * Wn;
err = zeros(size(tau_grid));
for k = 1:numel(tau_grid)
    A = JtW * ops.Jp + Q / tau_grid(k)^2;
    X = (A \ JtW) * (Yval - bae.mu_e);
    err(k) = mean(sqrt(sum((X - XTval).^2, 1)) ./ max(sqrt(sum(XTval.^2, 1)), eps));
end
[~, kb] = min(err);
if kb == 1 || kb == numel(tau_grid)
    warning('bae_build: best tau at the edge of the grid (%g).', tau_grid(kb));
end
A = JtW * ops.Jp + Q / tau_grid(kb)^2; A = (A + A') / 2;
Sigma = A \ eye(ops.P);
bae.G = Sigma * JtW;
bae.s_post = sqrt(max(diag(Sigma), 0));
bae.Sigma = Sigma;
bae.tau = tau_grid(kb);
fprintf('> BAE baseline: tau = %g (validation error %.3f); model error rms %.3g vs noise %.3g\n', ...
    bae.tau, err(kb), sqrt(mean(Ec(:).^2)), sigma_m);
end
