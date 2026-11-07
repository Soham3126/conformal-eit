function D = gen_dataset(sim, ops, cfg, N, seed, sigma_m, cache_file)
% GEN_DATASET  N random phantoms from model `sim`, with white measurement
% noise of FIXED standard deviation sigma_m (instrument noise, not SNR-scaled).
%   D.Y  M x N noisy data, D.Yc noise-free, D.XT P x N truth, D.T logical masks
if nargin >= 7 && ~isempty(cache_file) && exist(cache_file, 'file')
    S = load(cache_file, 'D');
    if S.D.N == N && S.D.seed == seed && abs(S.D.sigma_m - sigma_m) <= 1e-12 * max(sigma_m, 1)
        D = S.D; fprintf('> Loaded dataset %s\n', cache_file); return;
    end
end
rs = RandStream('mt19937ar', 'Seed', seed);
rn = RandStream('mt19937ar', 'Seed', seed + 1);
M = ops.M; P = ops.P;
D.Yc = zeros(M, N); D.XT = zeros(P, N); D.tg = cell(1, N);
t0 = tic;
for i = 1:N
    [D.Yc(:, i), D.XT(:, i), D.tg{i}] = sim_sample(sim, ops.C, cfg, rs);
    if mod(i, 50) == 0
        fprintf('   %d/%d phantoms (%.0fs)\n', i, N, toc(t0));
    end
end
D.Y = D.Yc + sigma_m * randn(rn, M, N);
D.T = D.XT ~= 0;
keep = any(D.T, 1);                       % drop phantoms too small for the grid
if any(~keep)
    fprintf('   dropped %d phantoms whose inclusions cover no pixel centre\n', sum(~keep));
end
D.Y = D.Y(:, keep); D.Yc = D.Yc(:, keep); D.XT = D.XT(:, keep);
D.T = D.T(:, keep); D.tg = D.tg(keep);
D.N = N; D.n = sum(keep); D.seed = seed; D.sigma_m = sigma_m; D.desc = sim.desc;
if nargin >= 7 && ~isempty(cache_file)
    d = fileparts(cache_file); if ~isempty(d) && ~exist(d, 'dir'), mkdir(d); end
    save(cache_file, 'D', '-v7.3');
end
end
