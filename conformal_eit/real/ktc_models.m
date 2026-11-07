function [fm_rec, fm_sim] = ktc_models(K, cfg) %#ok<INUSD>
% KTC_MODELS  Reconstruction and (finer) simulation models of the KTC tank.
% EIDORS' GREIT builder needs a 3D model, so the 2D tank is modelled as a
% cylinder (radius 1, height 1) with the 32 electrodes in one ring at z = 0,
% using the KTC injection (Inj) and measurement (Mpat) patterns.
fm_rec = cyl(8, 11, K);      % 32 boundary nodes per ring -> one per electrode
fm_sim = cyl(16, 21, K);     % finer: 64 boundary nodes per ring
v = fwd_solve(mk_image(fm_rec, 1));
if numel(v.meas) ~= numel(K.Uelref)
    error(['Model gives %d measurements but the data has %d. Check the ' ...
        'orientation of Mpat/Inj in the ref file.'], numel(v.meas), numel(K.Uelref));
end
end

function fm = cyl(rings, nz, K)
z = linspace(-0.5, 0.5, nz);
e_layer = (nz + 1) / 2;
p = mk_circ_tank(rings, z, {'planes', 32, e_layer});
p.solve = 'eidors_default'; p.system_mat = 'eidors_default';
p.jacobian = 'eidors_default'; p.normalize_measurements = 0;
Mp = K.Mpat; if size(Mp, 1) ~= 32, Mp = Mp'; end     % want 32 x Nmeas
Inj = K.Inj; if size(Inj, 1) ~= 32, Inj = Inj'; end  % want 32 x Ninj
st = struct('stimulation', {}, 'stim_pattern', {}, 'meas_pattern', {});
for k = 1:size(Inj, 2)
    st(k).stimulation = 'Amp';
    st(k).stim_pattern = sparse(Inj(:, k));
    st(k).meas_pattern = sparse(Mp');
end
p.stimulation = st;
fm = eidors_obj('fwd_model', p);
end
