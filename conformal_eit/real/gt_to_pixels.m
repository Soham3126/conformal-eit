function m = gt_to_pixels(G, C, ang_deg, flip)
% GT_TO_PIXELS  Sample a square ground-truth image (circle inscribed) at the
% reconstruction pixel centres C (unit disk), after rotating by ang_deg and
% optionally mirroring. Returns a logical anomaly mask (label > 0).
x = C(:, 1); y = C(:, 2);
if flip, x = -x; end
a = ang_deg * pi / 180;
xr = cos(a) * x - sin(a) * y; yr = sin(a) * x + cos(a) * y;
[nr, nc] = size(G);
col = min(max(round((xr + 1) / 2 * (nc - 1)) + 1, 1), nc);
row = min(max(round((1 - yr) / 2 * (nr - 1)) + 1, 1), nr);
m = G(sub2ind([nr nc], row, col)) > 0;
end
