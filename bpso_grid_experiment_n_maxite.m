%% BPSO grid experiment: population size (n) x max iterations (maxite)
% Runs the same algorithm as inverse_design_using_bpso_with_TLfwdmodel.m
% without modifying that file. Loads the ONNX model once, sweeps a grid,
% and saves a summary table plus per-run .mat files and plots.
%
% Usage (MATLAB Online or local MATLAB with ONNX support):
%   cd to the pixelant folder containing TLfwdmodel.onnx
%   bpso_grid_experiment_n_maxite

clc;
close all;

%% --- Grid configuration (edit these) ---
% Default: 3 x 2 = 6 runs. At ~10 min/run on MATLAB Online, plan ~1 hour total.
% For a fast smoke test, use: n_values = 200; maxite_values = [10, 25];
n_values = [200, 500, 1000];
maxite_values = [25, 50];

% Set true to also run the README baseline point explicitly (may duplicate a grid cell)
include_baseline_explicit = true;
baseline_n = 1000;
baseline_maxite = 50;

% PSO settings (match original script defaults)
wmax = 0.9;
wmin = 0.4;
c1 = 2;
c2 = 2;
maxrun = 1;
target_freq_hz = 3.6e9;
pass_band_half_width = 4;   % bins on each side of target (single band)

% Output
results_dir = fullfile(pwd, 'bpso_grid_results');
save_per_run_plots = true;
save_per_run_mat = true;
rng_seed = 42;              % reproducible grid; set [] to leave RNG unchanged

%% --- Build experiment list (all n x maxite combinations) ---
[Ngrid, Igrid] = ndgrid(n_values, maxite_values);
pairs = [Ngrid(:), Igrid(:)];
if include_baseline_explicit
    pairs = unique([pairs; baseline_n, baseline_maxite], 'rows', 'stable');
end
num_experiments = size(pairs, 1);

fprintf('BPSO grid experiment: %d runs\n', num_experiments);
fprintf('Results folder: %s\n\n', results_dir);

if ~isfile(fullfile(pwd, 'TLfwdmodel.onnx'))
    error('TLfwdmodel.onnx not found in %s', pwd);
end

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

if ~isempty(rng_seed)
    rng(rng_seed);
end

%% --- Load surrogate once (not counted in per-run BPSO time) ---
fprintf('Loading TLfwdmodel.onnx ...\n');
model_load_tic = tic;
net = importONNXNetwork('TLfwdmodel.onnx');
model_load_sec = toc(model_load_tic);
fprintf('Model loaded in %.2f s\n\n', model_load_sec);

%% --- Frequency / fitness setup (single band, same as original script) ---
m = 144;
num_pixels = 12;
fmin = 1e9;
fmax = 5e9;
N = 81;
freq = linspace(fmin, fmax, N);
center_fiu = find(freq == target_freq_hz);
if isempty(center_fiu)
    error('Target frequency %.4f GHz not on freq grid.', target_freq_hz / 1e9);
end
pass_band = center_fiu - pass_band_half_width : center_fiu + pass_band_half_width;
stop_band = [1 : center_fiu - pass_band_half_width - 1, ...
             center_fiu + pass_band_half_width + 1 : N];
pass_freq = freq(pass_band);
stop_freq = freq(stop_band);

%% --- Preallocate summary table ---
run_id = (1:num_experiments)';
summary = table( ...
    run_id, ...
    pairs(:, 1), pairs(:, 2), ...
    nan(num_experiments, 1), nan(num_experiments, 1), nan(num_experiments, 1), ...
    nan(num_experiments, 1), nan(num_experiments, 1), nan(num_experiments, 1), ...
    nan(num_experiments, 1), ...
    'VariableNames', { ...
    'run_id', 'n', 'maxite', ...
    'bpso_elapsed_sec', 'total_elapsed_sec', 'bestfun', ...
    'approx_evals', 'plateau_iteration', 'best_particle_index', ...
    'model_load_sec'});

all_runs = cell(num_experiments, 1);
grid_start_tic = tic;

%% --- Grid loop ---
for exp_idx = 1:num_experiments
    n = pairs(exp_idx, 1);
    maxite = pairs(exp_idx, 2);
    approx_evals = n + n * maxite;

    fprintf('=== Run %d/%d: n=%d, maxite=%d (~%d evals) ===\n', ...
        exp_idx, num_experiments, n, maxite, approx_evals);

    run_result = run_bpso_single( ...
        net, n, maxite, m, num_pixels, freq, pass_freq, stop_freq, ...
        wmax, wmin, c1, c2, maxrun, false);

    summary.run_id(exp_idx) = exp_idx;
    summary.n(exp_idx) = n;
    summary.maxite(exp_idx) = maxite;
    summary.bpso_elapsed_sec(exp_idx) = run_result.bpso_elapsed_sec;
    summary.total_elapsed_sec(exp_idx) = run_result.total_elapsed_sec;
    summary.bestfun(exp_idx) = run_result.bestfun;
    summary.approx_evals(exp_idx) = approx_evals;
    summary.plateau_iteration(exp_idx) = run_result.plateau_iteration;
    summary.best_particle_index(exp_idx) = run_result.best_particle_index_final;
    summary.model_load_sec(exp_idx) = model_load_sec;

    run_result.run_id = exp_idx;
    run_result.target_freq_hz = target_freq_hz;
    run_result.center_fiu = center_fiu;
    run_result.pass_band = pass_band;
    run_result.stop_band = stop_band;
    run_result.approx_evals = approx_evals;
    all_runs{exp_idx} = run_result;

    fprintf('  bpso_elapsed_sec = %.2f\n', run_result.bpso_elapsed_sec);
    fprintf('  bestfun          = %.4f\n', run_result.bestfun);
    fprintf('  plateau_iter     = %d\n\n', run_result.plateau_iteration);

    run_tag = sprintf('n%d_maxite%d', n, maxite);
    if save_per_run_mat
        run_mat_path = fullfile(results_dir, ['run_' run_tag '.mat']);
        save(run_mat_path, 'run_result', '-v7.3');
    end

    if save_per_run_plots
        fig_conv = figure('Visible', 'off');
        plot(run_result.ffmin, '-k', 'LineWidth', 1.2);
        xlabel('Iteration');
        ylabel('Fitness function value');
        title(sprintf('PSO convergence (n=%d, maxite=%d)', n, maxite));
        grid on;
        saveas(fig_conv, fullfile(results_dir, ['convergence_' run_tag '.png']));
        close(fig_conv);

        fig_s11 = figure('Visible', 'off');
        plot(freq, run_result.output_new, 'LineWidth', 1.2);
        xlabel('Frequency (Hz)');
        ylabel('Return loss');
        title(sprintf('Predicted S11 (n=%d, maxite=%d, bestfun=%.2f)', ...
            n, maxite, run_result.bestfun));
        grid on;
        saveas(fig_s11, fullfile(results_dir, ['s11_' run_tag '.png']));
        close(fig_s11);
    end
end

grid_total_sec = toc(grid_start_tic);

%% --- Save combined outputs ---
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
summary_csv = fullfile(results_dir, ['bpso_grid_summary_' timestamp '.csv']);
summary_mat = fullfile(results_dir, ['bpso_grid_summary_' timestamp '.mat']);
writetable(summary, summary_csv);
save(summary_mat, 'summary', 'all_runs', 'pairs', 'model_load_sec', ...
    'grid_total_sec', 'target_freq_hz', 'pass_band_half_width', 'rng_seed', '-v7.3');

fprintf('Grid finished in %.2f s (all runs + saves)\n', grid_total_sec);
fprintf('Summary CSV: %s\n', summary_csv);
fprintf('Summary MAT: %s\n', summary_mat);
disp(summary);

%% --- Summary figures across the grid ---
try
    fig_time = figure('Name', 'BPSO grid: time vs parameters');
    scatter(summary.n, summary.bpso_elapsed_sec, 80, summary.maxite, 'filled');
    xlabel('Population size n');
    ylabel('BPSO elapsed time (s)');
    title('BPSO time cost (first-toc equivalent per run)');
    colorbar;
    grid on;
    saveas(fig_time, fullfile(results_dir, 'grid_time_scatter.png'));

    fig_fitness = figure('Name', 'BPSO grid: fitness vs parameters');
    scatter(summary.n, summary.bestfun, 80, summary.maxite, 'filled');
    xlabel('Population size n');
    ylabel('bestfun');
    title('Final fitness vs n (color = maxite)');
    colorbar;
    grid on;
    saveas(fig_fitness, fullfile(results_dir, 'grid_fitness_scatter.png'));

    uniq_maxite = unique(summary.maxite);
    fig_scaling = figure('Name', 'BPSO grid: time scaling');
    hold on;
    for k = 1:numel(uniq_maxite)
        mask = summary.maxite == uniq_maxite(k);
        [ns, ord] = sort(summary.n(mask));
        plot(ns, summary.bpso_elapsed_sec(mask), '-o', ...
            'DisplayName', sprintf('maxite=%d', uniq_maxite(k)));
    end
    hold off;
    xlabel('Population size n');
    ylabel('BPSO elapsed time (s)');
    title('Time scaling by maxite');
    legend('Location', 'best');
    grid on;
    saveas(fig_scaling, fullfile(results_dir, 'grid_time_scaling.png'));
catch ME
    warning('Could not save summary figures: %s', ME.message);
end

fprintf('\nDone. Download the folder ''bpso_grid_results'' for your Mac.\n');

%% ========================================================================
function result = run_bpso_single(net, n, maxite, m, num_pixels, freq, ...
    pass_freq, stop_freq, wmax, wmin, c1, c2, maxrun, verbose)
%RUN_BPSO_SINGLE One BPSO run matching inverse_design_using_bpso_with_TLfwdmodel.m

    total_tic = tic;
    ffmin = nan(maxite, maxrun);
    ffite = zeros(maxrun, 1);
    fff = nan(maxrun, 1);
    rgbest = nan(maxrun, m);
    best_particle_index_final = nan;

    bpso_tic = tic;
    for run = 1:maxrun
        x = randi([0, 1], n, m);
        initial_inputmatrix = x;
        v = 0.1 * initial_inputmatrix;
        calc_r = zeros(n, numel(freq));
        Error_vec = zeros(n, 1);

        for i = 1:n
            X_t = reshape(initial_inputmatrix(i, :), num_pixels, num_pixels);
            X_t(6:7, 1) = 1;
            Ypr1 = predict(net, X_t);
            for jj = 1:numel(freq)
                if abs(Ypr1(1, jj)) > 10
                    calc_r(i, jj) = 10;
                elseif abs(Ypr1(1, jj)) < 5
                    calc_r(i, jj) = 0;
                else
                    calc_r(i, jj) = abs(Ypr1(1, jj));
                end
            end
            Error_vec(i, 1) = calculatefit(Ypr1(1, :), pass_freq, stop_freq, calc_r(i, :), freq);
        end

        [fmin0, index0] = max(Error_vec);
        pbest = initial_inputmatrix;
        gbest = initial_inputmatrix(index0, :);
        best_particle_index_final = index0;

        ite = 1;
        while ite <= maxite
            w = wmax - (wmax - wmin) * ite / maxite;
            for i = 1:n
                for j = 1:m
                    if (gbest(j) == pbest(i, j)) && (pbest(i, j) == 1)
                        v(i, j) = w * v(i, j) + c1 * rand() + c2 * rand();
                    elseif (gbest(j) == pbest(i, j)) && (pbest(i, j) == 0)
                        v(i, j) = w * v(i, j) - c1 * rand() - c2 * rand();
                    else
                        v(i, j) = w * v(i, j);
                    end
                end
            end

            v_prob = 1 ./ (1 + exp(-v));
            x_probs = rand(n, m);
            xx = double(x_probs < v_prob);

            f = zeros(n, 1);
            for i = 1:n
                tt = reshape(xx(i, :), num_pixels, num_pixels);
                tt(6:7, 1) = 1;
                Ypr2 = predict(net, tt);
                for jj = 1:numel(freq)
                    if abs(Ypr2(1, jj)) > 10
                        calc_r(i, jj) = 10;
                    elseif abs(Ypr2(1, jj)) < 5
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
            ffmin(ite, run) = fmin;
            ffite(run) = ite;
            if fmin > fmin0
                gbest = pbest(index, :);
                fmin0 = fmin;
                best_particle_index_final = index;
            end

            if verbose && ite == 1
                fprintf('%8s %8s %8s\n', 'Iteration', 'Best particle', 'Objective fun');
            end
            if verbose
                fprintf('%8d %8d %8.4f\n', ite, index, fmin0);
            end
            ite = ite + 1;
        end

        antenna_de = reshape(gbest, num_pixels, num_pixels);
        output_ne = predict(net, antenna_de);
        calc_r_final = zeros(1, numel(freq));
        for jj = 1:numel(freq)
            if abs(output_ne(1, jj)) > 10
                calc_r_final(jj) = 10;
            elseif abs(output_ne(1, jj)) < 5
                calc_r_final(jj) = 0;
            else
                calc_r_final(jj) = abs(output_ne(1, jj));
            end
        end
        fff(run) = calculatefit(output_ne(1, :), pass_freq, stop_freq, calc_r_final, freq);
        rgbest(run, :) = gbest;
    end

    result.bpso_elapsed_sec = toc(bpso_tic);

    [bestfun, bestrun] = min(fff);
    best_variables = rgbest(bestrun, :);
    antenna_des = reshape(best_variables, num_pixels, num_pixels);
    output_new = predict(net, antenna_des);
    if size(output_new, 1) > 1
        output_new = output_new(1, :);
    end

    result.total_elapsed_sec = toc(total_tic);
    result.bestfun = bestfun;
    result.bestrun = bestrun;
    result.best_variables = best_variables;
    result.antenna_des = antenna_des;
    result.output_new = output_new;
    result.ffmin = ffmin(1:ffite(bestrun), bestrun);
    result.ffite = ffite(bestrun);
    result.best_particle_index_final = best_particle_index_final;
    result.plateau_iteration = find_plateau_iteration(result.ffmin, bestfun);
end

function fit_value = calculatefit(~, p_freq, s_freq, calc_r, freq)
    tot_p = 0;
    tot_s = 0;
    for ii = 1:numel(p_freq)
        freq_index = find(freq == p_freq(ii));
        tot_p = tot_p + calc_r(freq_index);
    end
    for ii = 1:numel(s_freq)
        freq_index = find(freq == s_freq(ii));
        tot_s = tot_s + (10 - calc_r(freq_index));
    end
    fit_value = tot_p + tot_s;
end

function plateau_iter = find_plateau_iteration(ffmin_vec, bestfun)
    plateau_iter = numel(ffmin_vec);
    if isempty(ffmin_vec)
        return;
    end
    tol = max(1e-6, 1e-4 * abs(bestfun));
    idx = find(abs(ffmin_vec - bestfun) <= tol, 1, 'first');
    if ~isempty(idx)
        plateau_iter = idx;
    end
end
