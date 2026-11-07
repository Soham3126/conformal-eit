function [cal, tst] = split_indices(N, cal_frac, nsplit, seed)
% SPLIT_INDICES  Random calibration / test splits (one row per split).
rs = RandStream('mt19937ar', 'Seed', seed);
nc = round(cal_frac * N);
cal = zeros(nsplit, nc); tst = zeros(nsplit, N - nc);
for s = 1:nsplit
    p = randperm(rs, N);
    cal(s, :) = p(1:nc); tst(s, :) = p(nc + 1:end);
end
end
