%% BPSO warm-start experiment - post-processing and comparison plots
% Reads the compute-only outputs produced by bpso_warmstart_experiment.m for
% each mode (random / mixed / warm) and produces:
%   warmstart_convergence_comparison.png - mean convergence curve per mode
%   warmstart_final_quality.png          - final bestfun per repeat, per mode
%   warmstart_runtime.png                - BPSO wall-clock time per repeat, per mode
%   warmstart_summary.csv                - side-by-side metrics table
%
% This script does NOT run BPSO. Run it after the experiment(s) finish.
% It handles missing modes gracefully (plots whatever exists).
%
% Usage:
%   run bpso_warmstart_plots.m  (from anywhere; paths anchored to script folder)

clc;
close all;

%% -------------------- Settings --------------------
modes         = {'random', 'mixed', 'warm'};   % modes to look for
mode_colors   = {[0 0 0], [0 0.45 0.74], [0.85 0.33 0.10]};  % k, blue, orange
threshold     = 800;    % must match the experiment's threshold for the line
fail_floor    = 780;    % final_bestfun below this counts as a "failed" run
fitness_ceiling = 810;  % theoretical max of the fitness metric (reference line)

%% -------------------- Paths --------------------
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
base_dir = fullfile(script_dir, 'warmstart_results');
out_summary = fullfile(base_dir, 'warmstart_summary.csv');
out_conv    = fullfile(base_dir, 'warmstart_convergence_comparison.png');
out_quality = fullfile(base_dir, 'warmstart_final_quality.png');
out_runtime = fullfile(base_dir, 'warmstart_runtime.png');

if ~exist(base_dir, 'dir')
    error('No warmstart_results folder at %s. Run bpso_warmstart_experiment.m first.', base_dir);
end

%% -------------------- Load each mode's data --------------------
data = struct('mode', {}, 'color', {}, 'curves', {}, 'final', {}, ...
    'init', {}, 'itt', {}, 'elapsed', {});

for mi = 1:numel(modes)
    mode_str = modes{mi};
    mdir = fullfile(base_dir, mode_str);
    files = dir(fullfile(mdir, sprintf('curve_%s_rep*.mat', mode_str)));
    if isempty(files)
        fprintf('No data for mode "%s" (skipping).\n', mode_str);
        continue;
    end

    curves = [];
    finals = [];
    inits  = [];
    itts   = [];
    elapsed = [];
    for k = 1:numel(files)
        S = load(fullfile(files(k).folder, files(k).name));
        curves = [curves; S.conv_curve(:)']; %#ok<AGROW>
        finals = [finals; S.final_bestfun];  %#ok<AGROW>
        inits  = [inits;  S.init_bestfun];   %#ok<AGROW>
        itts   = [itts;   S.iters_to_threshold]; %#ok<AGROW>
        if isfield(S, 'elapsed_sec')
            elapsed = [elapsed; S.elapsed_sec]; %#ok<AGROW>
        else
            elapsed = [elapsed; NaN]; %#ok<AGROW>
        end
    end

    d.mode   = mode_str;
    d.color  = mode_colors{mi};
    d.curves = curves;
    d.final  = finals;
    d.init   = inits;
    d.itt    = itts;
    d.elapsed = elapsed;
    data(end+1) = d; %#ok<AGROW>
    fprintf('Loaded mode "%s": %d repeats.\n', mode_str, numel(files));
end

if isempty(data)
    error('No per-repeat curve files found under %s.', base_dir);
end

%% -------------------- Plot 1: mean convergence curves --------------------
fig1 = figure('Visible', 'off', 'Position', [100 100 760 480]);
hold on;
legend_entries = {};
for di = 1:numel(data)
    d = data(di);
    L = size(d.curves, 2);
    iters = 0:(L - 1);                 % iteration 0 = initial swarm
    mean_curve = mean(d.curves, 1);
    plot(iters, mean_curve, '-', 'Color', d.color, 'LineWidth', 2);
    legend_entries{end+1} = sprintf('%s (n_{rep}=%d)', d.mode, size(d.curves, 1)); %#ok<AGROW>
end
yline(threshold, '--', sprintf('threshold = %g', threshold), ...
    'Color', [0.4 0.4 0.4]);
yline(fitness_ceiling, ':', sprintf('ceiling = %g', fitness_ceiling), ...
    'Color', [0.6 0.6 0.6]);
hold off;
grid on;
xlabel('Iteration (0 = initial swarm)');
ylabel('Best fitness so far');
title('Warm-start vs random: mean convergence');
legend(legend_entries, 'Location', 'southeast');
saveas(fig1, out_conv);
close(fig1);
fprintf('Saved %s\n', out_conv);

%% -------------------- Plot 2: final quality per repeat --------------------
fig2 = figure('Visible', 'off', 'Position', [100 100 640 480]);
hold on;
xt = [];
xtl = {};
for di = 1:numel(data)
    d = data(di);
    xj = di + 0.08 * randn(numel(d.final), 1);   % small jitter for visibility
    scatter(xj, d.final, 45, d.color, 'filled', 'MarkerFaceAlpha', 0.7);
    plot([di-0.2 di+0.2], [mean(d.final) mean(d.final)], '-', ...
        'Color', d.color, 'LineWidth', 2);        % mean line
    xt(end+1) = di; %#ok<AGROW>
    xtl{end+1} = d.mode; %#ok<AGROW>
end
yline(fitness_ceiling, ':', sprintf('ceiling = %g', fitness_ceiling), ...
    'Color', [0.6 0.6 0.6]);
yline(fail_floor, '--', sprintf('fail floor = %g', fail_floor), ...
    'Color', [0.85 0.2 0.2]);
hold off;
grid on;
xlim([0.5, numel(data) + 0.5]);
set(gca, 'XTick', xt, 'XTickLabel', xtl);
ylabel('Final best fitness');
title('Final design quality per repeat (bar = mean)');
saveas(fig2, out_quality);
close(fig2);
fprintf('Saved %s\n', out_quality);

%% -------------------- Plot 3: runtime per repeat --------------------
% elapsed_sec comes from bpso_warmstart_experiment.m and measures BPSO compute
% only (tic/toc around the optimisation loop). All modes use the same n and
% maxite, so wall-clock time is expected to be similar: warm-start improves
% iterations-to-target, not total runtime, unless you add early stopping.
fig3 = figure('Visible', 'off', 'Position', [100 100 640 480]);
hold on;
xt = [];
xtl = {};
for di = 1:numel(data)
    d = data(di);
    xj = di + 0.08 * randn(numel(d.elapsed), 1);   % small jitter for visibility
    scatter(xj, d.elapsed, 45, d.color, 'filled', 'MarkerFaceAlpha', 0.7);
    plot([di-0.2 di+0.2], [mean(d.elapsed, 'omitnan') mean(d.elapsed, 'omitnan')], '-', ...
        'Color', d.color, 'LineWidth', 2);        % mean line
    xt(end+1) = di; %#ok<AGROW>
    xtl{end+1} = d.mode; %#ok<AGROW>
end
hold off;
grid on;
xlim([0.5, numel(data) + 0.5]);
set(gca, 'XTick', xt, 'XTickLabel', xtl);
ylabel('BPSO elapsed time (s)');
title('Runtime per repeat (bar = mean; same n and maxite for all modes)');
saveas(fig3, out_runtime);
close(fig3);
fprintf('Saved %s\n', out_runtime);

%% -------------------- Optional: fold in the 10-run baseline --------------------
% The original baseline (10 random runs) has final fitness values but no saved
% convergence curves; include its quality/reliability stats if present.
baseline_csv = fullfile(script_dir, 'bpso_grid_repeats_good_results', 'bpso_grid_results.csv');
baseline_final = [];
baseline_elapsed = [];
if isfile(baseline_csv)
    Tb = readtable(baseline_csv);
    if any(strcmp('bestfun', Tb.Properties.VariableNames))
        baseline_final = Tb.bestfun;
        fprintf('Included baseline (10-run) quality data from %s\n', baseline_csv);
    end
    if any(strcmp('bpso_elapsed_sec', Tb.Properties.VariableNames))
        baseline_elapsed = Tb.bpso_elapsed_sec;
    end
end

%% -------------------- Summary CSV --------------------
rows = table();
for di = 1:numel(data)
    d = data(di);
    itt = d.itt;
    row = table(string(d.mode), size(d.curves, 1), ...
        mean(d.init), mean(d.final), std(d.final), min(d.final), ...
        mean(itt(~isnan(itt))), median(itt(~isnan(itt))), sum(~isnan(itt)), ...
        sum(d.final < fail_floor), ...
        mean(d.elapsed, 'omitnan'), std(d.elapsed, 'omitnan'), ...
        'VariableNames', {'mode', 'num_repeats', 'mean_init_bestfun', ...
        'mean_final_bestfun', 'std_final_bestfun', 'min_final_bestfun', ...
        'mean_iters_to_threshold', 'median_iters_to_threshold', ...
        'runs_reaching_threshold', 'num_failures', ...
        'mean_elapsed_sec', 'std_elapsed_sec'});
    rows = [rows; row]; %#ok<AGROW>
end

if ~isempty(baseline_final)
    row = table("random_baseline10", numel(baseline_final), ...
        NaN, mean(baseline_final), std(baseline_final), min(baseline_final), ...
        NaN, NaN, NaN, sum(baseline_final < fail_floor), ...
        mean(baseline_elapsed, 'omitnan'), std(baseline_elapsed, 'omitnan'), ...
        'VariableNames', rows.Properties.VariableNames);
    rows = [rows; row];
end

writetable(rows, out_summary);
fprintf('Saved %s\n', out_summary);

fprintf('\n==== Warm-start comparison summary ====\n');
disp(rows);
fprintf(['Read as: lower mean_iters_to_threshold = faster convergence; ' ...
    'higher min_final_bestfun and fewer num_failures = more reliable; ' ...
    'mean_final_bestfun near %g = maintained quality; ' ...
    'mean_elapsed_sec is wall-clock BPSO time (similar across modes because ' ...
    'all runs use the same maxite; warm-start saves iterations, not runtime).\n'], fitness_ceiling);
