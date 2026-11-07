function [lam, idx, bound] = crc_lambda(Lcal, grid, alpha, B, direction)
% CRC_LAMBDA  Conformal risk control (Angelopoulos et al., ICLR 2024).
%
%   Lcal      n x K matrix: loss of each calibration sample at each grid value
%   grid      1 x K increasing grid of lambda values
%   alpha     target risk level
%   B         upper bound on the loss (losses must lie in [0, B])
%   direction 'nonincreasing' : loss falls as lambda grows (e.g. spurious area,
%                               interval miscoverage)  -> smallest valid lambda
%             'nondecreasing' : loss grows with lambda (e.g. missed target
%                               fraction)              -> largest valid lambda
%
% Guarantee (exchangeable calibration and test samples, monotone loss):
%     E[ L_test(lam) ] <= alpha
% using the finite-sample correction  (n * Rhat + B) / (n + 1) <= alpha.
n = size(Lcal, 1);
Rhat = mean(Lcal, 1);
bound = (n * Rhat + B) / (n + 1);
ok = find(bound <= alpha);
switch direction
    case 'nonincreasing'
        if isempty(ok), idx = numel(grid); else, idx = ok(1); end
    case 'nondecreasing'
        if isempty(ok), idx = 1; else, idx = ok(end); end
    otherwise
        error('crc_lambda: unknown direction %s', direction);
end
lam = grid(idx);
end
