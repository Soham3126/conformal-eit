function ops = ops_build_generic(fmdl, cache_file, greit_radius, greit_opt)
% OPS_BUILD_GENERIC  Reconstruction operators for ANY EIDORS forward model
% (2D or 3D). Cached because GREIT training and the Jacobian are slow.
%   ops.R   P x M   GREIT reconstruction matrix
%   ops.Jp  M x P   pixel Jacobian  J_fine * C_vol (exact volume fractions)
%   ops.C   P x 2   pixel centres, ops.dx pitch
%   ops.Dg  E x P   sparse pixel graph gradient (4-neighbour)
%   ops.sig_px      sqrt(diag(R R')): GREIT image noise std per unit meas. noise
if nargin < 3 || isempty(greit_radius), greit_radius = 0.2; end
if nargin < 4 || isempty(greit_opt)
    greit_opt = struct('img_size', [32 32], 'distr', 3, 'noise_figure', 0.5, 'target_size', 0.05);
end
if ~isempty(cache_file) && exist(cache_file, 'file')
    S = load(cache_file, 'ops'); ops = S.ops;
    fprintf('> Loaded operators from %s\n', cache_file); return;
end
fprintf('> Building operators (slow, cached afterwards)...\n');
img = mk_image(fmdl, 1.0);
imdl = mk_GREIT_model(img, greit_radius, 5, greit_opt);
R = imdl.solve_use_matrix.RM;
[P, M] = size(R);

rec = imdl.rec_model; Nt = size(rec.elems, 1);
tc = zeros(Nt, 2);
for k = 1:Nt, tc(k, :) = mean(rec.nodes(rec.elems(k, :), 1:2), 1); end
if Nt == 2 * P, C = (tc(1:2:end, :) + tc(2:2:end, :)) / 2; else, C = tc(1:P, :); end
dx = 2 / greit_opt.img_size(1);

xyzr = [C(:, 1)'; C(:, 2)'; (dx / sqrt(pi)) * ones(1, P)];
[C_vol, failed] = mk_c2f_circ_mapping(fmdl, xyzr);
if any(failed), warning('mk_c2f_circ_mapping: %d pixels failed', sum(failed)); end
Jp = full(calc_jacobian(img) * C_vol);

D2 = (C(:, 1) - C(:, 1)').^2 + (C(:, 2) - C(:, 2)').^2;
[ii, jj] = find(triu(D2 > 0 & D2 <= (1.05 * dx)^2));
E = numel(ii);
Dg = sparse([1:E, 1:E]', [ii; jj], [ones(E, 1); -ones(E, 1)], E, P);

ops = struct('R', R, 'Jp', Jp, 'C', C, 'dx', dx, 'Dg', Dg, 'P', P, 'M', M, ...
    'sig_px', sqrt(sum(R.^2, 2)), 'fmdl', fmdl, 'img_bkgnd', img);
if ~isempty(cache_file)
    d = fileparts(cache_file); if ~isempty(d) && ~exist(d, 'dir'), mkdir(d); end
    save(cache_file, 'ops', '-v7.3');
end
fprintf('> Operators ready: P = %d pixels, M = %d measurements\n', P, M);
end
