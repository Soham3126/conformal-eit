function lam = crc_interval_lambda(Rn, G, alpha)
% CRC_INTERVAL_LAMBDA  Exact conformal risk control for interval miscoverage
% (no grid). Loss of sample i at lambda:
%     L_i(lambda) = (1/|G_i|) * sum_{p in G_i} 1{ Rn(p,i) > lambda }   (0 if G_i empty)
% Returns the smallest lambda with (n*Rhat(lambda) + 1)/(n + 1) <= alpha, which
% guarantees E[L_test(lambda)] <= alpha for an exchangeable test sample.
%   Rn  P x n normalised residuals |x - xhat| ./ scale
%   G   P x n logical group masks (true(P,n) = all pixels)
n = size(Rn, 2);
t = (alpha * (n + 1) - 1) / n;            % allowed weighted exceedance mass
cnt = sum(G, 1);
W = G ./ max(cnt, 1) / n;                 % weight of each pixel residual
r = Rn(G); w = W(G);
if isempty(r), lam = 0; return; end
if t < 0, lam = max(r); return; end       % too few samples for this alpha
[rs, o] = sort(r, 'descend');
cw = cumsum(w(o));
k = find(cw <= t + 1e-15, 1, 'last');     % residuals allowed to exceed lambda
if isempty(k), lam = rs(1);
elseif k >= numel(rs), lam = 0;
else, lam = rs(k + 1);
end
end
