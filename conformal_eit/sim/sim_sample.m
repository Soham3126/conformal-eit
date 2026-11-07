function [y, xt, tg] = sim_sample(sim, C, cfg, rs)
% SIM_SAMPLE  One random phantom: 1..cfg.n_incl_max non-overlapping inclusions
% (spheres centred at z = 0 in 3D, disks in 2D), each conductive or resistive.
%   y   M x 1 noise-free difference data  v_incl - v_homog
%   xt  P x 1 true conductivity change on the pixel grid
%   tg  K x 4 rows [x y r dsigma]
K = randi(rs, cfg.n_incl_max);
tg = zeros(0, 4);
tries = 0;
while size(tg, 1) < K && tries < 500
    tries = tries + 1;
    r = cfg.r_range(1) + diff(cfg.r_range) * rand(rs);
    rho = (cfg.max_rad - r) * sqrt(rand(rs)); th = 2 * pi * rand(rs);
    c = [rho * cos(th), rho * sin(th)];
    if ~isempty(tg) && any(sqrt(sum((tg(:, 1:2) - c).^2, 2)) < tg(:, 3) + r + 0.05), continue; end
    if rand(rs) < cfg.p_negative
        ds = cfg.contrast_neg(1) + diff(cfg.contrast_neg) * rand(rs);
    else
        ds = cfg.contrast_pos(1) + diff(cfg.contrast_pos) * rand(rs);
    end
    tg(end + 1, :) = [c, r, ds]; %#ok<AGROW>
end
img = sim.img_bg;
xt = zeros(size(C, 1), 1);
for k = 1:size(tg, 1)
    d2 = (sim.ec(:, 1) - tg(k, 1)).^2 + (sim.ec(:, 2) - tg(k, 2)).^2;
    if size(sim.ec, 2) >= 3 && ~(isfield(cfg, 'cylinder') && cfg.cylinder)
        d2 = d2 + sim.ec(:, 3).^2;            % sphere; cylinders span the full height
    end
    img.elem_data(d2 <= tg(k, 3)^2) = 1 + tg(k, 4);
    xt((C(:, 1) - tg(k, 1)).^2 + (C(:, 2) - tg(k, 2)).^2 <= tg(k, 3)^2) = tg(k, 4);
end
v = fwd_solve(img);
y = v.meas - sim.vh;
end
