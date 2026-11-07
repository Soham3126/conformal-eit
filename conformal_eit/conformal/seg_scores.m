function S = seg_scores(X, kind, scale)
% SEG_SCORES  Anomaly score maps from reconstructions (columns of X).
%   'rel' : |x| / max|x|           the Graz-consensus normalisation (generic)
%   'snr' : |x| ./ scale           physics-informed: amplitude in units of the
%                                  reconstruction's own noise std per pixel
%                                  (GREIT: sigma_m*sqrt(diag(RR')),
%                                   Bayes: posterior std)
switch kind
    case 'rel'
        S = abs(X) ./ max(max(abs(X), [], 1), eps);
    case 'snr'
        S = abs(X) ./ max(scale(:), eps);
    otherwise
        error('seg_scores: unknown kind %s', kind);
end
end
