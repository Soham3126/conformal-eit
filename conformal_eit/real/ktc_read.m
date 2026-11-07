function K = ktc_read(folder)
% KTC_READ  Read Kuopio Tomography Challenge 2023 data placed in `folder`:
%   ref*.mat            reference (water) measurement: Uelref, Inj (or Injref), Mpat
%   data<N>.mat         one phantom each: Uel
%   true<N>.mat / .png  ground-truth segmentation (0 water, 1 resistive, 2 conductive)
% Variable names are detected by prefix, so small naming differences are tolerated.
% Phantom N is paired with true<N> by the number in the file name.
if ~exist(folder, 'dir'), error('KTC folder not found: %s', folder); end
rf = dir(fullfile(folder, 'ref*.mat'));
if isempty(rf), error('No ref*.mat in %s', folder); end
R = load(fullfile(folder, rf(1).name));
K.Uelref = pick(R, {'Uelref', 'Uel'}, rf(1).name);
try
    K.Inj = pick(R, {'Injref', 'Inj', 'CurrentPattern'}, rf(1).name);
catch                                   % some releases keep Inj in the data files
    df0 = dir(fullfile(folder, 'data*.mat'));
    K.Inj = pick(load(fullfile(folder, df0(1).name)), {'Inj'}, df0(1).name);
end
try
    K.Mpat = pick(R, {'Mpat', 'MeasPattern'}, rf(1).name);
catch
    df0 = dir(fullfile(folder, 'data*.mat'));
    K.Mpat = pick(load(fullfile(folder, df0(1).name)), {'Mpat'}, df0(1).name);
end
df = dir(fullfile(folder, 'data*.mat'));
if isempty(df), error('No data*.mat files in %s', folder); end
K.name = {}; K.Uel = []; K.gt = {};
for i = 1:numel(df)
    num = regexp(df(i).name, '\d+', 'match', 'once');
    g = [dir(fullfile(folder, ['true' num '.mat'])); dir(fullfile(folder, ['true' num '.png']))];
    if isempty(g)
        fprintf('   (no ground truth for %s, skipped)\n', df(i).name); continue;
    end
    S = load(fullfile(folder, df(i).name));
    u = pick(S, {'Uel'}, df(i).name);
    if numel(u) ~= numel(K.Uelref)
        error('%s has %d voltages but the reference has %d.', df(i).name, numel(u), numel(K.Uelref));
    end
    K.Uel(:, end + 1) = u(:);
    K.gt{end + 1} = read_gt(fullfile(folder, g(1).name));
    K.name{end + 1} = df(i).name;
end
fprintf('> KTC: %d phantoms with ground truth, %d voltages each, %d injections\n', ...
    numel(K.name), numel(K.Uelref), size(K.Inj, 2));
end

function v = pick(S, prefixes, fname)
f = fieldnames(S);
for p = 1:numel(prefixes)
    m = find(strncmpi(f, prefixes{p}, numel(prefixes{p})), 1);
    if ~isempty(m), v = double(S.(f{m})); return; end
end
error('None of {%s} found in %s. Variables present: %s', strjoin(prefixes, ', '), ...
    fname, strjoin(f', ', '));
end

function G = read_gt(file)
[~, ~, ext] = fileparts(file);
if strcmpi(ext, '.png')
    I = imread(file); if ndims(I) == 3, I = I(:, :, 1); end
    u = unique(I(:));                     % map grey levels to labels 0,1,2
    G = zeros(size(I));
    for k = 2:numel(u), G(I == u(k)) = k - 1; end
else
    S = load(file); f = fieldnames(S);
    G = [];
    for k = 1:numel(f)
        v = S.(f{k});
        if isnumeric(v) || islogical(v)
            if ismatrix(v) && min(size(v)) > 16, G = double(v); break; end
        end
    end
    if isempty(G), error('No 2D image found in %s', file); end
end
end
