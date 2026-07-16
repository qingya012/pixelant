%% BPSO warm-start + early-stopping experiment - post-processing and comparison plots
% Combines two result trees produced by two different scripts:
%   warmstart_results/random/            - bpso_warmstart_experiment.m,
%                                           full 50-iteration runs, no early
%                                           stopping (used as-is, the reference
%                                           baseline; early stopping was found
%                                           to be unreliable for "random").
%   warmstart_earlystop_results/{mixed,warm}/ - bpso_warmstart_early_stopping.m,
%                                           same seeds as the original
%                                           mixed/warm runs, but the loop can
%                                           stop early on a fitness plateau.
%
% Produces (in warmstart_earlystop_results/):
%   earlystop_convergence_comparison.png - per-repeat curves (each ending at
%                                           its REAL stop) + a forward-filled
%                                           mean curve per mode
%   earlystop_iterations_run.png         - actual_iterations per repeat, per
%                                           mode (the headline early-stopping
%                                           result: how much shorter did the
%                                           run actually go)
%   earlystop_final_quality.png          - final bestfun per repeat, per mode
%   earlystop_runtime.png                - BPSO wall-clock time per repeat,
%                                           per mode (this is where the real
%                                           speedup from early stopping shows
%                                           up, unlike the no-early-stopping
%                                           comparison which had similar
%                                           runtimes across modes by design)
%   earlystop_summary.csv                - side-by-side metrics table,
%                                           including speedup_vs_random
%
% conv_curve length differs across repeats/modes on purpose (true stop point,
% not padded when saved) - all alignment for plotting/averaging happens HERE,
% at plot time, via forward-fill (repeat the last value out to the longest
% curve being compared). Nothing on disk is modified.
%
% This script does NOT run BPSO. Run it after bpso_warmstart_early_stopping.m
% has produced "mixed" and "warm" results (random is read from the original
% warmstart_results/ tree).
%
% Usage:
%   run bpso_warmstart_earlystop_plots.m   (from anywhere; paths anchored to
%   the script folder)

clc;
close all;

%% -------------------- Settings --------------------
% source: 'baseline'  -> read from warmstart_results/<mode>/      (no early stopping)
%         'earlystop' -> read from warmstart_earlystop_results/<mode>/ (early stopping)
mode_specs = struct( ...
    'mode',   {'random',        'mixed',      'warm'}, ...
    'source', {'baseline',      'earlystop',  'earlystop'}, ...
    'color',  {[0 0 0],         [0 0.45 0.74], [0.85 0.33 0.10]});

threshold       = 800;   % must match both experiments' threshold for the line
fail_floor      = 780;   % final_bestfun below this counts as a "failed" run
fitness_ceiling = 810;   % theoretical max of the fitness metric (reference line)

%% -------------------- Paths --------------------
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
baseline_dir  = fullfile(script_dir, 'warmstart_results');
earlystop_dir = fullfile(script_dir, 'warmstart_earlystop_results');

if ~exist(earlystop_dir, 'dir')
    error(['No warmstart_earlystop_results folder at %s. ' ...
        'Run bpso_warmstart_early_stopping.m first.'], earlystop_dir);
end

out_conv    = fullfile(earlystop_dir, 'earlystop_convergence_comparison.png');
out_iters   = fullfile(earlystop_dir, 'earlystop_iterations_run.png');
out_quality = fullfile(earlystop_dir, 'earlystop_final_quality.png');
out_runtime = fullfile(earlystop_dir, 'earlystop_runtime.png');
out_summary = fullfile(earlystop_dir, 'earlystop_summary.csv');

%% -------------------- Load each mode's data --------------------
data = struct('mode', {}, 'color', {}, 'source', {}, 'curves', {}, ...
    'final', {}, 'init', {}, 'itt', {}, 'elapsed', {}, 'actual_iter', {}, ...
    'stop_reason', {});

for mi = 1:numel(mode_specs)
    spec = mode_specs(mi);
    mode_str = spec.mode;
    if strcmp(spec.source, 'baseline')
        mdir = fullfile(baseline_dir, mode_str);
    else
        mdir = fullfile(earlystop_dir, mode_str);
    end
    files = dir(fullfile(mdir, sprintf('curve_%s_rep*.mat', mode_str)));
    if isempty(files)
        fprintf('No data for mode "%s" in %s (skipping).\n', mode_str, mdir);
        continue;
    end

    curves      = {};
    finals      = [];
    inits       = [];
    itts        = [];
    elapsed     = [];
    actual_iter = [];
    stop_reason = {};
    for k = 1:numel(files)
        S = load(fullfile(files(k).folder, files(k).name));
        curves{end+1} = S.conv_curve(:)'; %#ok<AGROW>
        finals  = [finals;  S.final_bestfun];  %#ok<AGROW>
        inits   = [inits;   S.init_bestfun];   %#ok<AGROW>
        itts    = [itts;    S.iters_to_threshold]; %#ok<AGROW>
        if isfield(S, 'elapsed_sec')
            elapsed = [elapsed; S.elapsed_sec]; %#ok<AGROW>
        else
            elapsed = [elapsed; NaN]; %#ok<AGROW>
        end
        % Baseline (random) runs predate the actual_iterations/stop_reason
        % fields - they always ran the full maxite with no early stopping.
        if isfield(S, 'actual_iterations')
            actual_iter = [actual_iter; S.actual_iterations]; %#ok<AGROW>
        else
            actual_iter = [actual_iter; S.maxite]; %#ok<AGROW>
        end
        if isfield(S, 'stop_reason')
            stop_reason{end+1} = char(S.stop_reason); %#ok<AGROW>
        else
            stop_reason{end+1} = 'max_iterations'; %#ok<AGROW>
        end
    end

    d.mode        = mode_str;
    d.color       = spec.color;
    d.source      = spec.source;
    d.curves      = curves;         % cell array, one row vector per repeat (ragged lengths OK)
    d.final       = finals;
    d.init        = inits;
    d.itt         = itts;
    d.elapsed     = elapsed;
    d.actual_iter = actual_iter;
    d.stop_reason = stop_reason;
    data(end+1) = d; %#ok<AGROW>
    fprintf('Loaded mode "%s" (%s): %d repeats.\n', mode_str, spec.source, numel(files));
end

if isempty(data)
    error('No per-repeat curve files found under %s or %s.', baseline_dir, earlystop_dir);
end

%% -------------------- Plot 1: convergence, real stops + forward-filled mean --------------------
% Individual curves are plotted EXACTLY as saved (no padding) so each one
% visibly ends at its real stopping iteration. The mean curve per mode is
% computed by forward-filling (holding the last value) each repeat's curve
% out to that mode's longest repeat, purely for this plot - the underlying
% data on disk is untouched.
fig1 = figure('Visible', 'off', 'Position', [100 100 780 500]);
hold on;
legend_handles = [];
legend_entries = {};
for di = 1:numel(data)
    d = data(di);
    n_rep = numel(d.curves);

    % Thin, low-alpha individual curves with a marker at the real stop point.
    % (Line transparency needs to be set on the object after creation - MATLAB
    % does not reliably accept a 4-element RGBA 'Color' at plot() call time.)
    for k = 1:n_rep
        c = d.curves{k};
        iters = 0:(numel(c) - 1);
        hline = plot(iters, c, '-', 'Color', d.color, 'LineWidth', 1, ...
            'HandleVisibility', 'off');
        hline.Color(4) = 0.35;
        plot(iters(end), c(end), 'o', 'Color', d.color, ...
            'MarkerFaceColor', d.color, 'MarkerSize', 4, ...
            'HandleVisibility', 'off');
    end

    % Forward-filled mean curve (bold).
    Lmax = max(cellfun(@numel, d.curves));
    stacked = nan(n_rep, Lmax);
    for k = 1:n_rep
        c = d.curves{k};
        stacked(k, :) = [c, repmat(c(end), 1, Lmax - numel(c))];
    end
    mean_curve = mean(stacked, 1);
    h = plot(0:(Lmax - 1), mean_curve, '-', 'Color', d.color, 'LineWidth', 2.5);
    legend_handles(end+1) = h; %#ok<AGROW>
    legend_entries{end+1} = sprintf('%s (n_{rep}=%d, mean of forward-filled curves)', ...
        d.mode, n_rep); %#ok<AGROW>
end
yline(threshold, '--', sprintf('threshold = %g', threshold), 'Color', [0.4 0.4 0.4]);
yline(fitness_ceiling, ':', sprintf('ceiling = %g', fitness_ceiling), 'Color', [0.6 0.6 0.6]);
hold off;
grid on;
xlabel('Iteration (0 = initial swarm)');
ylabel('Best fitness so far');
title('Warm-start + early stopping: convergence (dots = real stop, per repeat)');
legend(legend_handles, legend_entries, 'Location', 'southeast');
saveas(fig1, out_conv);
close(fig1);
fprintf('Saved %s\n', out_conv);

%% -------------------- Plot 2: iterations actually run --------------------
% The headline early-stopping result: how many of the 50 budgeted iterations
% each run actually needed before the plateau rule (or maxite) stopped it.
fig2 = figure('Visible', 'off', 'Position', [100 100 640 480]);
hold on;
xt = [];
xtl = {};
for di = 1:numel(data)
    d = data(di);
    xj = di + 0.08 * randn(numel(d.actual_iter), 1);
    scatter(xj, d.actual_iter, 45, d.color, 'filled', 'MarkerFaceAlpha', 0.7);
    plot([di-0.2 di+0.2], [mean(d.actual_iter) mean(d.actual_iter)], '-', ...
        'Color', d.color, 'LineWidth', 2);
    xt(end+1) = di; %#ok<AGROW>
    xtl{end+1} = d.mode; %#ok<AGROW>
end
yline(50, ':', 'maxite = 50', 'Color', [0.6 0.6 0.6]);
hold off;
grid on;
xlim([0.5, numel(data) + 0.5]);
ylim([0, 53]);
set(gca, 'XTick', xt, 'XTickLabel', xtl);
ylabel('Iterations actually run (actual\_iterations)');
title('Iterations run before stopping (bar = mean)');
saveas(fig2, out_iters);
close(fig2);
fprintf('Saved %s\n', out_iters);

%% -------------------- Plot 3: final quality per repeat --------------------
fig3 = figure('Visible', 'off', 'Position', [100 100 640 480]);
hold on;
xt = [];
xtl = {};
for di = 1:numel(data)
    d = data(di);
    xj = di + 0.08 * randn(numel(d.final), 1);
    scatter(xj, d.final, 45, d.color, 'filled', 'MarkerFaceAlpha', 0.7);
    plot([di-0.2 di+0.2], [mean(d.final) mean(d.final)], '-', ...
        'Color', d.color, 'LineWidth', 2);
    xt(end+1) = di; %#ok<AGROW>
    xtl{end+1} = d.mode; %#ok<AGROW>
end
yline(fitness_ceiling, ':', sprintf('ceiling = %g', fitness_ceiling), 'Color', [0.6 0.6 0.6]);
yline(fail_floor, '--', sprintf('fail floor = %g', fail_floor), 'Color', [0.85 0.2 0.2]);
hold off;
grid on;
xlim([0.5, numel(data) + 0.5]);
set(gca, 'XTick', xt, 'XTickLabel', xtl);
ylabel('Final best fitness');
title('Final design quality per repeat (bar = mean)');
saveas(fig3, out_quality);
close(fig3);
fprintf('Saved %s\n', out_quality);

%% -------------------- Plot 4: runtime per repeat --------------------
% Unlike the no-early-stopping comparison (where runtime was similar across
% modes by construction), this is where early stopping should show a real
% wall-clock speedup for mixed/warm relative to random.
fig4 = figure('Visible', 'off', 'Position', [100 100 640 480]);
hold on;
xt = [];
xtl = {};
for di = 1:numel(data)
    d = data(di);
    xj = di + 0.08 * randn(numel(d.elapsed), 1);
    scatter(xj, d.elapsed, 45, d.color, 'filled', 'MarkerFaceAlpha', 0.7);
    plot([di-0.2 di+0.2], [mean(d.elapsed, 'omitnan') mean(d.elapsed, 'omitnan')], '-', ...
        'Color', d.color, 'LineWidth', 2);
    xt(end+1) = di; %#ok<AGROW>
    xtl{end+1} = d.mode; %#ok<AGROW>
end
hold off;
grid on;
xlim([0.5, numel(data) + 0.5]);
set(gca, 'XTick', xt, 'XTickLabel', xtl);
ylabel('BPSO elapsed time (s)');
title('Runtime per repeat (bar = mean; early stopping saves wall-clock time)');
saveas(fig4, out_runtime);
close(fig4);
fprintf('Saved %s\n', out_runtime);

%% -------------------- Summary CSV --------------------
random_idx = find(strcmp({data.mode}, 'random'), 1);
if ~isempty(random_idx)
    mean_elapsed_random = mean(data(random_idx).elapsed, 'omitnan');
else
    mean_elapsed_random = NaN;
end

rows = table();
for di = 1:numel(data)
    d = data(di);
    itt = d.itt;
    n_plateau = sum(strcmp(d.stop_reason, 'plateau'));
    mean_elapsed = mean(d.elapsed, 'omitnan');
    speedup_vs_random = mean_elapsed_random / mean_elapsed;
    row = table(string(d.mode), string(d.source), numel(d.curves), ...
        mean(d.init), mean(d.final), std(d.final), min(d.final), ...
        mean(itt(~isnan(itt))), median(itt(~isnan(itt))), sum(~isnan(itt)), ...
        sum(d.final < fail_floor), ...
        mean(d.actual_iter), n_plateau, ...
        mean_elapsed, std(d.elapsed, 'omitnan'), speedup_vs_random, ...
        'VariableNames', {'mode', 'source', 'num_repeats', 'mean_init_bestfun', ...
        'mean_final_bestfun', 'std_final_bestfun', 'min_final_bestfun', ...
        'mean_iters_to_threshold', 'median_iters_to_threshold', ...
        'runs_reaching_threshold', 'num_failures', ...
        'mean_actual_iterations', 'num_plateau_stops', ...
        'mean_elapsed_sec', 'std_elapsed_sec', 'speedup_vs_random'});
    rows = [rows; row]; %#ok<AGROW>
end

writetable(rows, out_summary);
fprintf('Saved %s\n', out_summary);

fprintf('\n==== Warm-start + early-stopping comparison summary ====\n');
disp(rows);
fprintf(['Read as: mean_actual_iterations << maxite (50) and num_plateau_stops > 0 means ' ...
    'the plateau rule is doing real work; speedup_vs_random compares mean wall-clock time ' ...
    'against the (never early-stopped) random baseline; mean_final_bestfun near %g and ' ...
    'few num_failures means quality was preserved despite stopping early.\n'], fitness_ceiling);
