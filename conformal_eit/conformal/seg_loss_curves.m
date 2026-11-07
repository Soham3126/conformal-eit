function [Lin, Lout, Area] = seg_loss_curves(S, T, grid)
% SEG_LOSS_CURVES  Segmentation losses of the set {p : score_p >= lambda}
% for every lambda in grid, for many samples at once.
%
%   S     P x N score maps (non-negative)
%   T     P x N logical true anomaly masks (each column non-empty)
%   grid  1 x K increasing thresholds
%
%   Lin(i,k)  = min( |S_k \ T| / |T| , 1 )   spurious area relative to true area
%               (non-increasing in lambda)   -> "inner" set guarantee
%   Lout(i,k) = |T \ S_k| / |T|              missed fraction of the target
%               (non-decreasing in lambda)   -> "outer" set guarantee
%   Area(i,k) = |S_k| / |T|                  set size relative to true area
N = size(S, 2); K = numel(grid);
Lin = zeros(N, K); Lout = zeros(N, K); Area = zeros(N, K);
for i = 1:N
    s = S(:, i); t = T(:, i); nt = max(sum(t), 1);
    in = s >= grid;                                   % P x K
    Lin(i, :)  = min(sum(in & ~t, 1) / nt, 1);
    Lout(i, :) = sum(~in & t, 1) / nt;
    Area(i, :) = sum(in, 1) / nt;
end
end
