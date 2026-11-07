% init_env.m - Initializes project paths and EIDORS environment
clc;
fprintf('Initializing conformal EIT environment...\n');

project_root = fileparts(mfilename('fullpath'));

eidors_dir = fullfile(project_root, 'lib', 'eidors');
eidors_startup_file = fullfile(eidors_dir, 'eidors_startup.m');

if exist(eidors_startup_file, 'file')
    current_dir = pwd;
    cd(eidors_dir);
    eidors_startup;
    cd(current_dir);
    
    % Ensure overloads root is present for spdiag, but strip octave subfolder
    addpath(fullfile(eidors_dir, 'overloads'));
    octave_dir = fullfile(eidors_dir, 'overloads', 'octave');
    if exist(octave_dir, 'dir')
        warning('off', 'MATLAB:rmpath:DirNotFound');
        rmpath(octave_dir);
        warning('on', 'MATLAB:rmpath:DirNotFound');
    end
    
    fprintf('> EIDORS loaded and paths sanitized successfully.\n');
else
    error('EIDORS startup not found at: %s', eidors_startup_file);
end

fprintf('> Environment ready.\n');