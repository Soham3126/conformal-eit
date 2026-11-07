function sim = make_sim_3d(level, ops, cfg)
% MAKE_SIM_3D  Data-generating model with a controlled amount of model error
% relative to the reconstruction model (ops.fmdl = EIDORS 'b3cr').
%   level 0 : the reconstruction mesh itself (inverse crime; only the prior
%             of the Bayesian baseline is wrong)
%   level 1 : finer mesh, same geometry
%   level 2 : level 1 + fixed random electrode position errors (cfg.jitter_deg)
%   level 3 : level 2 + tank radius error (cfg.radius_err, e.g. +3%)
if level == 0
    fm = ops.fmdl; desc = 'L0 same mesh (inverse crime)';
else
    z = linspace(-0.7, 0.7, 29);
    e_layer = find(abs(z) == min(abs(z)), 1);
    p = mk_circ_tank(16, z, {'planes', 16, e_layer});
    p.stimulation = ops.fmdl.stimulation;
    if isfield(ops.fmdl, 'meas_select'), p.meas_select = ops.fmdl.meas_select; end
    p.solve = 'eidors_default'; p.system_mat = 'eidors_default';
    p.jacobian = 'eidors_default'; p.normalize_measurements = 0;
    fm = eidors_obj('fwd_model', p);
    desc = 'L1 fine mesh';
    if level >= 2
        rs = RandStream('mt19937ar', 'Seed', cfg.seed + 77);
        for e = 1:numel(fm.electrode)
            d = cfg.jitter_deg * randn(rs) * pi / 180;
            nd = fm.electrode(e).nodes;
            xy = fm.nodes(nd, 1:2);
            fm.nodes(nd, 1:2) = xy * [cos(d) sin(d); -sin(d) cos(d)];
        end
        desc = sprintf('L2 fine mesh + electrode error (sd %.1f deg)', cfg.jitter_deg);
    end
    if level >= 3
        fm.nodes(:, 1:2) = fm.nodes(:, 1:2) * (1 + cfg.radius_err);
        desc = sprintf('L3 L2 + tank radius error %+.0f%%', 100 * cfg.radius_err);
    end
end
sim = finish_sim(fm, desc);
end

function sim = finish_sim(fm, desc)
sim.fmdl = fm; sim.desc = desc;
sim.img_bg = mk_image(fm, 1.0);
v = fwd_solve(sim.img_bg); sim.vh = v.meas;
n = fm.nodes; e = fm.elems;
sim.ec = zeros(size(e, 1), size(n, 2));
for k = 1:size(e, 2), sim.ec = sim.ec + n(e(:, k), :); end
sim.ec = sim.ec / size(e, 2);
fprintf('> Simulation model: %s (%d elements)\n', desc, size(e, 1));
end
