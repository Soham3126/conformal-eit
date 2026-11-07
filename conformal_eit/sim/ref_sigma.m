function sigma_m = ref_sigma(sim, ops, snr_db, ds)
% REF_SIGMA  Fixed instrument noise level: the std that gives `snr_db` for a
% reference inclusion (radius 0.15 at the centre, conductivity change ds).
if nargin < 4, ds = 1; end
img = sim.img_bg;
d2 = sim.ec(:, 1).^2 + sim.ec(:, 2).^2;
if size(sim.ec, 2) >= 3, d2 = d2 + sim.ec(:, 3).^2; end
img.elem_data(d2 <= 0.15^2) = 1 + ds;
v = fwd_solve(img);
y = v.meas - sim.vh;
sigma_m = sqrt(mean(y.^2)) * 10^(-snr_db / 20);
fprintf('> Measurement noise sigma = %.3g (%.0f dB on the reference target)\n', sigma_m, snr_db);
end
