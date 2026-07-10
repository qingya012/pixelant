%% BPSO warm-start experiment (compute-only, checkpointed, resumable)
% Studies whether seeding BPSO with a previously optimized ("good") layout
% improves convergence SPEED and RELIABILITY while keeping final quality.
%
% ONLY the population initialization differs between modes. Every other part
% of the BPSO (velocity update, sigmoid sampling, fitness, pbest/gbest,
% convergence tracking, stopping) is identical to
% inverse_design_using_bpso_with_TLfwdmodel.m / the baseline repeats script.
%
% Three initialization modes (set init_mode below, run once per mode):
%   "random" - 100% random particles (control; same as baseline)
%   "mixed"  - seed_fraction of particles are perturbed copies of good_layout,
%              the rest are random
%   "warm"   - 100% of particles are perturbed copies of good_layout
% Perturbation = flip ~flip_rate of the bits per seeded particle (keeps the
% seeded particles non-identical so the swarm still has diversity).
%
% Design choices for robustness (MATLAB can hang on very long jobs):
%   - COMPUTE ONLY. No plotting here (use bpso_warmstart_plots.m afterwards).
%   - Saves after EVERY repeat: one .mat per repeat + a metrics CSV.
%   - RESUMABLE: repeats already saved are skipped, so a crash just means
%     re-launching to continue.
%   - Per-iteration console printing is off by default (verbose=false), which
%     also avoids Command Window flooding that can stall the desktop.
%   - Never overwrites good_layout.mat (it is read-only input here).
%
% Outputs (in warmstart_results/<mode>/):
%   metrics_<mode>.csv          - one row per repeat (scalar metrics)
%   curve_<mode>_rep<k>.mat     - convergence curve + best layout for repeat k
%
% Usage:
%   1) Set init_mode = "random" / "mixed" / "warm"
%   2) Run this script (from anywhere; paths are anchored to the script folder)
%   3) Repeat for each mode. Run bpso_warmstart_plots.m to compare.

clc;
close all;

%% -------------------- User settings (edit these) --------------------
init_mode     = "warm";     % "random" | "mixed" | "warm"  (run once per mode)
num_repeats   = 5;          % repeats for this mode (seeded modes need few; 3-5)
flip_rate     = 0.05;       % fraction of bits flipped per seeded particle (~7 of 144)
seed_fraction = 0.10;       % fraction of seeded particles (only used by "mixed")
threshold     = 800;        % fitness target for the iterations-to-threshold metric

% Baseline BPSO settings (README defaults; do NOT change - keeps comparison fair)
n      = 1000;
maxite = 50;
wmax   = 0.9;
wmin   = 0.4;
c1     = 2;
c2     = 2;
maxrun = 1;

verbose = false;            % true -> print every iteration (noisy, can stall)

%% -------------------- Path setup (anchored to this file) --------------------
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

% Locate the ONNX model: prefer cwd, else repo root two levels up.
onnx_candidates = {fullfile(pwd, 'TLfwdmodel.onnx'), ...
    fullfile(script_dir, '..', '..', 'TLfwdmodel.onnx')};
onnx_path = '';
for k = 1:numel(onnx_candidates)
    if isfile(onnx_candidates{k})
        onnx_path = onnx_candidates{k};
        break;
    end
end
if isempty(onnx_path)
    error(['TLfwdmodel.onnx not found in the current folder or at the repo ' ...
        'root (%s). cd to the folder containing the model, then re-run.'], ...
        fullfile(script_dir, '..', '..'));
end

% Seed file produced by the baseline repeats run (read-only input).
seed_mat = fullfile(script_dir, 'bpso_grid_repeats_good_results', 'good_layout.mat');

% Per-mode results folder (compute outputs only).
mode_str   = char(init_mode);
results_dir = fullfile(script_dir, 'warmstart_results', mode_str);
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
metrics_csv = fullfile(results_dir, ['metrics_' mode_str '.csv']);

%% -------------------- Load seed layout if needed --------------------
good_row = [];
m = 144;                       % 12x12 pixels flattened
if any(strcmp(mode_str, {'mixed', 'warm'}))
    if ~isfile(seed_mat)
        error(['init_mode "%s" needs a seed layout but %s was not found. ' ...
            'Run the baseline first to create good_layout.mat.'], mode_str, seed_mat);
    end
    S = load(seed_mat, 'good_layout');
    if ~isfield(S, 'good_layout')
        error('%s does not contain a variable named good_layout.', seed_mat);
    end
    gl = double(S.good_layout);
    if numel(gl) ~= m
        error('good_layout has %d elements but expected m = %d.', numel(gl), m);
    end
    good_row = reshape(gl, 1, m);   % works for 12x12 or 1x144 or 144x1
end
good_row_logical = logical(good_row);

%% -------------------- Problem setup (same as baseline) --------------------
fprintf('Loading %s ...\n', onnx_path);
net = importONNXNetwork(onnx_path);

fmin_hz = 1e9;
fmax_hz = 5e9;
N = 81;
freq = linspace(fmin_hz, fmax_hz, N);
center_fiu = find(freq == 3.6e9);

pass_band = [center_fiu-4:center_fiu+4];
stop_band = [1:center_fiu-5, center_fiu+5:81];
pass_freq = freq(pass_band);
stop_freq = freq(stop_band);

% Seeds are disjoint from the baseline run (which used 42 + config*100 + rep).
switch mode_str
    case 'random', mode_seed_offset = 0;
    case 'mixed',  mode_seed_offset = 1000;
    case 'warm',   mode_seed_offset = 2000;
    otherwise
        error('Unknown init_mode "%s". Use "random", "mixed", or "warm".', mode_str);
end
rng_base = 9000;

%% -------------------- Resume: find repeats already done --------------------
% A repeat is "done" if its per-repeat .mat checkpoint exists on disk.
existing = dir(fullfile(results_dir, sprintf('curve_%s_rep*.mat', mode_str)));
done_reps = [];
for k = 1:numel(existing)
    tok = regexp(existing(k).name, 'rep(\d+)\.mat$', 'tokens', 'once');
    if ~isempty(tok)
        done_reps(end+1) = str2double(tok{1}); %#ok<AGROW>
    end
end

fprintf('\n==== Warm-start experiment: mode="%s", %d repeats, n=%d, maxite=%d ====\n', ...
    mode_str, num_repeats, n, maxite);
fprintf('Results folder: %s\n', results_dir);
if ~isempty(done_reps)
    fprintf('Resuming: %d repeat(s) already saved, will skip them.\n', numel(done_reps));
end

%% -------------------- Main loop over repeats --------------------
for repeat_id = 1:num_repeats
    curve_mat = fullfile(results_dir, sprintf('curve_%s_rep%d.mat', mode_str, repeat_id));

    if ismember(repeat_id, done_reps) && isfile(curve_mat)
        fprintf('  [skip] repeat %d already done.\n', repeat_id);
        continue;
    end

    seed = rng_base + mode_seed_offset + repeat_id;
    rng(seed);

    fprintf('\n---------- mode="%s"  repeat %d/%d  (seed=%d) ----------\n', ...
        mode_str, repeat_id, num_repeats, seed);

    tic;
    for run = 1:maxrun
        %% ============ INITIALIZATION (the only part that changes) ============
        switch mode_str
            case 'random'
                x = randi([0, 1], n, m);
            case 'mixed'
                n_seed = max(1, round(seed_fraction * n));
                x = randi([0, 1], n, m);              % rest are random
                for i = 1:n_seed                       % first n_seed are seeded
                    flip_mask = rand(1, m) < flip_rate;
                    x(i, :) = double(xor(good_row_logical, flip_mask));
                end
            case 'warm'
                x = zeros(n, m);
                for i = 1:n
                    flip_mask = rand(1, m) < flip_rate;
                    x(i, :) = double(xor(good_row_logical, flip_mask));
                end
        end
        initial_inputmatrix = x;
        %% ====================================================================

        v = 0.1 * initial_inputmatrix;
        Error_vec = -inf(n, 1);
        for i = 1:n
            X_t = reshape(initial_inputmatrix(i, :), 12, 12);
            X_t(6:7, 1) = 1;
            Ypr1 = predict(net, X_t);
            for jj = 1:length(freq)
                if (abs(Ypr1(1, jj)) > 10)
                    calc_r(i, jj) = 10;
                elseif (abs(Ypr1(1, jj)) < 5)
                    calc_r(i, jj) = 0;
                else
                    calc_r(i, jj) = abs(Ypr1(1, jj));
                end
            end
            Error_vec(i, 1) = calculatefit(Ypr1(1, :), pass_freq, stop_freq, calc_r(i, :), freq);
        end
        [fmin0, index0] = max(Error_vec(:, 1));
        pbest = initial_inputmatrix;
        gbest = initial_inputmatrix(index0, :);

        % Convergence curve including iteration 0 (initial swarm best).
        conv_curve = zeros(1, maxite + 1);
        conv_curve(1) = fmin0;

        ite = 1;
        while ite <= maxite
            w = wmax - (wmax - wmin) * ite / maxite;
            for i = 1:n
                for j = 1:m
                    if (gbest(j) == pbest(i, j) && pbest(i, j) == 1)
                        v(i, j) = w * v(i, j) + c1 * rand() + c2 * rand();
                    elseif (gbest(j) == pbest(i, j) && pbest(i, j) == 0)
                        v(i, j) = w * v(i, j) - c1 * rand() - c2 * rand();
                    else
                        v(i, j) = w * v(i, j);
                    end
                end
            end

            for i = 1:n
                for j = 1:m
                    v_prob(i, j) = 1 / (1 + exp(-v(i, j)));
                end
            end

            x_probs = rand(n, m);
            for i = 1:n
                for j = 1:m
                    if x_probs(i, j) < v_prob(i, j)
                        xx(i, j) = 1;
                    else
                        xx(i, j) = 0;
                    end
                end
            end

            for i = 1:n
                tt = reshape(xx(i, :), 12, 12);
                tt(6:7, 1) = 1;
                Ypr2 = predict(net, tt);
                for jj = 1:length(freq)
                    if (abs(Ypr2(1, jj)) > 10)
                        calc_r(i, jj) = 10;
                    elseif (abs(Ypr2(1, jj)) < 5)
                        calc_r(i, jj) = 0;
                    else
                        calc_r(i, jj) = abs(Ypr2(1, jj));
                    end
                end
                f(i, 1) = calculatefit(Ypr2(1, :), pass_freq, stop_freq, calc_r(i, :), freq);
            end

            for i = 1:n
                if f(i, 1) > Error_vec(i, 1)
                    pbest(i, :) = xx(i, :);
                    Error_vec(i, 1) = f(i, 1);
                end
            end
            [fmin, index] = max(Error_vec);
            conv_curve(ite + 1) = fmin;
            if fmin > fmin0
                gbest = pbest(index, :);
                fmin0 = fmin;
            end
            if verbose
                if ite == 1
                    disp('Iteration Best particle Objective fun');
                end
                disp(sprintf('%8g %8g %8.4f', ite, index, fmin0));
            end
            ite = ite + 1;
        end

        % Final re-evaluation of gbest (same metric as baseline "bestfun").
        antenna_de = reshape(gbest, 12, 12);
        antenna_de(6:7, 1) = 1;  % BUGFIX: force feed pixels (port line), matching in-loop scoring so final_bestfun matches the convergence curve
        output_ne = predict(net, antenna_de);
        for jj = 1:length(freq)
            if (abs(output_ne(1, jj)) > 10)
                calc_r(1, jj) = 10;
            elseif (abs(output_ne(1, jj)) < 5)
                calc_r(1, jj) = 0;
            else
                calc_r(1, jj) = abs(output_ne(1, jj));
            end
        end
        err = calculatefit(output_ne(1, :), pass_freq, stop_freq, calc_r(1, :), freq);
        fff(run) = err;
        rgbest(run, :) = gbest;
    end
    elapsed_sec = toc;

    [final_bestfun, bestrun] = max(fff);
    best_layout = rgbest(bestrun, :);
    best_layout([6, 7]) = 1;  % B-FIX: lock feed (port) pixels into the saved design vector so it is physically valid
    init_bestfun = conv_curve(1);

    % Iterations-to-threshold (0 = initial swarm already meets it; NaN = never).
    hit = find(conv_curve >= threshold, 1, 'first');
    if isempty(hit)
        iters_to_threshold = NaN;
    else
        iters_to_threshold = hit - 1;
    end

    % ---- Save this repeat immediately (checkpoint) ----
    save(curve_mat, 'conv_curve', 'best_layout', 'init_bestfun', ...
        'final_bestfun', 'iters_to_threshold', 'threshold', 'seed', ...
        'init_mode', 'flip_rate', 'seed_fraction', 'n', 'maxite', 'elapsed_sec');

    % Rebuild the metrics CSV from all checkpoints on disk (robust to resume).
    rebuild_metrics_csv(results_dir, mode_str, metrics_csv);

    if isnan(iters_to_threshold)
        itt_str = sprintf('never (<%g)', threshold);
    else
        itt_str = sprintf('%d', iters_to_threshold);
    end
    fprintf(['  done: init=%.2f  final=%.2f  iters_to_%g=%s  time=%.1f s\n' ...
        '  saved %s\n'], init_bestfun, final_bestfun, threshold, itt_str, ...
        elapsed_sec, curve_mat);
end

%% -------------------- Summary for this mode --------------------
fprintf('\n==== Mode "%s" complete ====\n', mode_str);
Tmode = rebuild_metrics_csv(results_dir, mode_str, metrics_csv);
if ~isempty(Tmode)
    fprintf('  repeats saved      : %d\n', height(Tmode));
    fprintf('  mean init_bestfun  : %.3f\n', mean(Tmode.init_bestfun));
    fprintf('  mean final_bestfun : %.3f (std %.3f)\n', ...
        mean(Tmode.final_bestfun), std(Tmode.final_bestfun));
    fprintf('  min  final_bestfun : %.3f  (reliability: higher is safer)\n', ...
        min(Tmode.final_bestfun));
    itt = Tmode.iters_to_threshold;
    fprintf('  iters_to_%g       : mean %.2f, median %.1f, reached %d/%d runs\n', ...
        threshold, mean(itt(~isnan(itt))), median(itt(~isnan(itt))), ...
        sum(~isnan(itt)), height(Tmode));
    fprintf('  metrics CSV        : %s\n', metrics_csv);
end
fprintf('Next: set init_mode to another mode and re-run, then run bpso_warmstart_plots.m\n');

%% -------------------- Local functions --------------------
% Rebuild the per-mode metrics CSV by scanning all per-repeat checkpoints.
% Keeps the CSV consistent with disk even across crashes/resumes.
function T = rebuild_metrics_csv(results_dir, mode_str, metrics_csv)
    files = dir(fullfile(results_dir, sprintf('curve_%s_rep*.mat', mode_str)));
    T = table();
    for k = 1:numel(files)
        S = load(fullfile(files(k).folder, files(k).name));
        tok = regexp(files(k).name, 'rep(\d+)\.mat$', 'tokens', 'once');
        rid = str2double(tok{1});
        if isfield(S, 'elapsed_sec'), es = S.elapsed_sec; else, es = NaN; end
        row = table(string(mode_str), S.n, S.maxite, rid, S.seed, ...
            S.init_bestfun, S.final_bestfun, S.iters_to_threshold, ...
            S.flip_rate, S.seed_fraction, es, ...
            'VariableNames', {'mode', 'n', 'maxite', 'repeat_id', 'seed', ...
            'init_bestfun', 'final_bestfun', 'iters_to_threshold', ...
            'flip_rate', 'seed_fraction', 'elapsed_sec'});
        T = [T; row]; %#ok<AGROW>
    end
    if ~isempty(T)
        T = sortrows(T, 'repeat_id');
        writetable(T, metrics_csv);
    end
end

% Fitness function (unchanged from baseline).
function fit_value = calculatefit(s_params, p_freq, s_freq, calc_r, freq)
    tot_p = 0;
    tot_s = 0;
    for ii = 1:length(p_freq)
        freq_index = find(freq == p_freq(ii));
        tot_p = tot_p + calc_r(freq_index);
    end
    for ii = 1:length(s_freq)
        freq_index = find(freq == s_freq(ii));
        tot_s = tot_s + (10 - calc_r(freq_index));
    end
    fit_value = tot_p + tot_s;
end
