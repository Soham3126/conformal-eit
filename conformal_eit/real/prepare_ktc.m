function prepare_ktc(src, dst)
% PREPARE_KTC  Flatten the KTC2023 download into the layout ktc_read expects.
%   src  folder containing Codes_Matlab/ and EvaluationData_full/
%   dst  output folder (conformal_eit/data/ktc)
% Each phantom is differenced against ITS OWN reference (training ref or its
% level's ref), then re-based on the training reference, so that
%   Uel_written - Uelref_training = Uel_phantom - Uelref_own.
% The level-1 reference is also written as a water-only phantom (data0 / true0,
% all zeros) so the noise level can be estimated from two water measurements.
if ~exist(dst, 'dir'), mkdir(dst); end
trn = fullfile(src, 'Codes_Matlab', 'TrainingData');
gtt = fullfile(src, 'Codes_Matlab', 'GroundTruths');
evd = fullfile(src, 'EvaluationData_full', 'evaluation_datasets');
evg = fullfile(src, 'EvaluationData_full', 'GroundTruths');

R0 = load(fullfile(trn, 'ref.mat'));
[u0, f0] = uel(R0, 'ref.mat');
copyfile(fullfile(trn, 'ref.mat'), fullfile(dst, 'ref.mat'));
n = 0;
% ---- training phantoms 1..4 ----
for i = 1:4
    n = n + 1;
    write_phantom(fullfile(trn, sprintf('data%d.mat', i)), u0, u0, dst, n);
    copyfile(fullfile(gtt, sprintf('true%d.mat', i)), fullfile(dst, sprintf('true%d.mat', n)));
end
% ---- evaluation phantoms, 7 levels x 3 ----
for L = 1:7
    Rl = load(fullfile(evd, sprintf('level%d', L), 'ref.mat'));
    ul = uel(Rl, sprintf('level%d/ref.mat', L));
    check_patterns(R0, Rl, L);
    if L == 1   % water-only phantom from two independent water measurements
        S = struct(f0, ul(:) - u0(:) + u0(:));
        save(fullfile(dst, 'data0.mat'), '-struct', 'S');
        G = load(fullfile(gtt, 'true1.mat')); gf = fieldnames(G);
        Z = struct(gf{1}, zeros(size(G.(gf{1}))));
        save(fullfile(dst, 'true0.mat'), '-struct', 'Z');
    end
    for k = 1:3
        n = n + 1;
        write_phantom(fullfile(evd, sprintf('level%d', L), sprintf('data%d.mat', k)), ul, u0, dst, n);
        copyfile(fullfile(evg, sprintf('level_%d', L), sprintf('%d_true.mat', k)), ...
            fullfile(dst, sprintf('true%d.mat', n)));
    end
end
fprintf('> Wrote %d phantoms + 1 water phantom to %s\n', n, dst);
end

function write_phantom(file, u_own, u0, dst, n)
S = load(file);
[u, f] = uel(S, file);
S.(f) = u(:) - u_own(:) + u0(:);
save(fullfile(dst, sprintf('data%d.mat', n)), '-struct', 'S');
end

function [u, f] = uel(S, name)
fn = fieldnames(S);
m = find(strncmpi(fn, 'Uel', 3), 1);
if isempty(m), error('No Uel* variable in %s (found: %s)', name, strjoin(fn', ', ')); end
f = fn{m}; u = double(S.(f));
end

function check_patterns(R0, Rl, L)
for p = {'Inj', 'Mpat'}
    a = pickp(R0, p{1}); b = pickp(Rl, p{1});
    if ~isempty(a) && ~isempty(b) && ~isequal(size(a), size(b))
        error('Level %d uses a different %s pattern from the training data.', L, p{1});
    end
    if ~isempty(a) && ~isempty(b) && max(abs(a(:) - b(:))) > 1e-9 * max(abs(a(:)))
        warning('Level %d %s differs from the training data; results may be affected.', L, p{1});
    end
end
end

function v = pickp(S, pre)
fn = fieldnames(S); m = find(strncmpi(fn, pre, numel(pre)), 1);
if isempty(m), v = []; else, v = double(S.(fn{m})); end
end
