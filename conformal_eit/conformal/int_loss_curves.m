function L = int_loss_curves(Rn, grid, G)
% INT_LOSS_CURVES  Interval miscoverage losses.
% Interval for pixel p:  xhat_p +- lambda * scale_p.
%   Rn    P x N normalised residuals |x_p - xhat_p| / scale_p
%   grid  1 x K increasing lambda values
%   G     (optional) P x N logical group mask; loss is computed inside the
%         group only (0 when the group is empty in that sample)
%   L(i,k) = fraction of (group) pixels NOT covered at lambda_k  (non-increasing)
N = size(Rn, 2); K = numel(grid);
if nargin < 3 || isempty(G), G = true(size(Rn)); end
L = zeros(N, K);
for i = 1:N
    r = Rn(G(:, i), i);
    if isempty(r), continue; end
    L(i, :) = mean(r > grid, 1);
end
end
