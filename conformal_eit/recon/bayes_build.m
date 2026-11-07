function bay = bayes_build(ops, sigma_m, tau, eps_reg)
% BAYES_BUILD  Linear-Gaussian Bayesian reconstruction (the standard EIT
% uncertainty baseline) on the pixel basis.
%   likelihood  y | x ~ N(Jp x, sigma_m^2 I)
%   prior       x ~ N(0, tau^2 Q^-1),  Q = Dg'Dg + eps*I  (smoothness prior)
%   posterior   x | y ~ N(G y, Sigma)
%       Sigma = (Jp'Jp/sigma_m^2 + Q/tau^2)^-1,   G = Sigma Jp' / sigma_m^2
% bay.s_post = sqrt(diag(Sigma)) gives the Bayesian 1-sigma pixel error bar.
if nargin < 4 || isempty(eps_reg), eps_reg = 1e-3; end
Q = full(ops.Dg' * ops.Dg) + eps_reg * eye(ops.P);
A = (ops.Jp' * ops.Jp) / sigma_m^2 + Q / tau^2;
A = (A + A') / 2;
Sigma = A \ eye(ops.P);
bay.G = Sigma * ops.Jp' / sigma_m^2;
bay.s_post = sqrt(max(diag(Sigma), 0));
bay.Sigma = Sigma;
bay.tau = tau; bay.sigma_m = sigma_m;
end
