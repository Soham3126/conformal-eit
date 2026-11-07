function X = tv_recon(Y, ops, sigma_m, mu, opt)
% TV_RECON  Total-variation regularised linear reconstruction, column by column:
%   x = argmin 0.5/sigma_m^2 ||Jp x - y||^2 + mu * ||Dg x||_1
% Solved with diagonally preconditioned Chambolle-Pock (Pock & Chambolle 2011).
% No sign constraint, because phantoms may mix conductive and resistive inclusions.
if nargin < 5, opt = struct(); end
maxit = getf(opt, 'maxit', 2000); tol = getf(opt, 'tol', 1e-4);
K = ops.Jp / sigma_m; Kt = K'; Dg = ops.Dg; Dt = Dg';
tau = 1 ./ max(sum(abs(K), 1)' + full(sum(abs(Dg), 1))', eps);
s1 = 1 ./ max(sum(abs(K), 2), eps); s2 = 1 ./ max(full(sum(abs(Dg), 2)), eps);
N = size(Y, 2); X = zeros(ops.P, N);
for i = 1:N
    d = Y(:, i) / sigma_m;
    x = zeros(ops.P, 1); xb = x; y1 = zeros(size(K, 1), 1); y2 = zeros(size(Dg, 1), 1);
    for it = 1:maxit
        y1 = (y1 + s1 .* (K * xb - d)) ./ (1 + s1);
        y2 = min(max(y2 + s2 .* (Dg * xb), -mu), mu);
        xn = x - tau .* (Kt * y1 + Dt * y2);
        xb = 2 * xn - x;
        if mod(it, 50) == 0 && norm(xn - x) <= tol * max(norm(xn), eps), x = xn; break; end
        x = xn;
    end
    X(:, i) = x;
end
end

function v = getf(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
