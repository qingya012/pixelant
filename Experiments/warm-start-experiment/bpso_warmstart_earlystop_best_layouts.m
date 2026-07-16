%% BPSO warm-start + early-stopping experiment - retrieve best final layouts (post-processing only)
% Scans the per-repeat checkpoints produced by bpso_warmstart_early_stopping.m
% for "mixed" and "warm" (random is not run with early stopping - see that
% script for why), picks the repeat with the highest final_bestfun per mode,
% and pulls out its best_layout (the 12x12 pixel design, feed pixels already
% locked to 1).
%
% This does NOT re-run BPSO; it only reads curve_<mode>_rep*.mat files that
% already contain 'best_layout' (see bpso_warmstart_early_stopping.m).
%
% Also loads the original seed ("good") layout from
% bpso_grid_repeats_good_results/good_layout.mat, so it can be compared
% side-by-side against the best mixed/warm layouts, and used as the reference
% for Hamming distance / similarity (mixed vs. warm are NOT compared against
% each other - only each mode's best layout vs. the good/seed layout).
%
% Outputs (in warmstart_earlystop_results/):
%   earlystop_best_layouts.mat            - struct array 'best_layouts', one
%                                            entry per mode (mixed, warm),
%                                            with fields: mode, repeat_id,
%                                            seed, init_bestfun, final_bestfun,
%                                            actual_iterations, stop_reason,
%                                            layout_12x12, layout_row (1x144)
%   earlystop_best_layouts_summary.csv    - one row per mode (+ good/seed):
%                                            which repeat won and its scores
%   earlystop_best_layouts_grid.png       - side-by-side pixel-grid image:
%                                            good/seed, mixed, warm, in a row
%   earlystop_best_layouts_hamming_vs_good.csv - Hamming distance / percent-
%                                            identical of each mode's best
%                                            layout vs. the good/seed layout
%                                            only (feed pixels excluded)
% Outputs (in warmstart_earlystop_results/<mode>/):
%   best_layout_<mode>.png                - pixel-grid image of that mode's
%                                            best layout
%
% Usage:
%   run bpso_warmstart_earlystop_best_layouts.m   (from anywhere; paths
%   anchored to the script folder). Run after bpso_warmstart_early_stopping.m
%   has produced results for "mixed" and "warm".

clc;
close all;

%% -------------------- Settings --------------------
modes = {'mixed', 'warm'};   % random excluded on purpose - not run with early stopping

%% -------------------- Path setup (anchored to this file) --------------------
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

base_dir = fullfile(script_dir, 'warmstart_earlystop_results');
if ~exist(base_dir, 'dir')
    error(['No warmstart_earlystop_results folder at %s. ' ...
        'Run bpso_warmstart_early_stopping.m first.'], base_dir);
end

out_mat     = fullfile(base_dir, 'earlystop_best_layouts.mat');
out_summary = fullfile(base_dir, 'earlystop_best_layouts_summary.csv');
out_grid    = fullfile(base_dir, 'earlystop_best_layouts_grid.png');
out_hamming = fullfile(base_dir, 'earlystop_best_layouts_hamming_vs_good.csv');

good_dir = fullfile(script_dir, 'bpso_grid_repeats_good_results');
good_mat = fullfile(good_dir, 'good_layout.mat');

%% -------------------- Find the winning repeat per mode --------------------
% Each best_layouts(i) is built with struct(...) in one call (rather than
% incremental d.field = value assignments into a possibly stale variable),
% so it always has exactly this field set - avoids the classic MATLAB
% "subscripted assignment between dissimilar structures" error if some other
% script left a same-named variable with different fields in the workspace.
best_layouts = struct('mode', {}, 'repeat_id', {}, 'seed', {}, ...
    'init_bestfun', {}, 'final_bestfun', {}, 'actual_iterations', {}, ...
    'stop_reason', {}, 'layout_12x12', {}, 'layout_row', {});

for mi = 1:numel(modes)
    mode_str = modes{mi};
    mdir = fullfile(base_dir, mode_str);
    files = dir(fullfile(mdir, sprintf('curve_%s_rep*.mat', mode_str)));
    if isempty(files)
        fprintf('No data for mode "%s" (skipping).\n', mode_str);
        continue;
    end

    best_val = -inf;
    best_S   = [];
    best_rid = NaN;

    for k = 1:numel(files)
        S = load(fullfile(files(k).folder, files(k).name));
        tok = regexp(files(k).name, 'rep(\d+)\.mat$', 'tokens', 'once');
        rid = str2double(tok{1});
        if isfield(S, 'final_bestfun') && S.final_bestfun > best_val
            best_val = S.final_bestfun;
            best_S   = S;
            best_rid = rid;
        end
    end

    if isempty(best_S)
        fprintf('Mode "%s": no repeat had a usable final_bestfun (skipping).\n', mode_str);
        continue;
    end

    if isfield(best_S, 'actual_iterations'), ai = best_S.actual_iterations; else, ai = NaN; end
    if isfield(best_S, 'stop_reason'), sr = string(best_S.stop_reason); else, sr = "unknown"; end

    layout_row = double(best_S.best_layout(:)');
    d = struct('mode', mode_str, 'repeat_id', best_rid, 'seed', best_S.seed, ...
        'init_bestfun', best_S.init_bestfun, 'final_bestfun', best_S.final_bestfun, ...
        'actual_iterations', ai, 'stop_reason', sr, ...
        'layout_12x12', reshape(layout_row, 12, 12), 'layout_row', layout_row);
    best_layouts(end+1) = d; %#ok<AGROW>

    fprintf('Mode "%s": best repeat %d (seed=%d, init=%.2f, final=%.2f, iters=%d, stop=%s)\n', ...
        mode_str, best_rid, d.seed, d.init_bestfun, d.final_bestfun, ai, sr);
end

if isempty(best_layouts)
    error('No per-repeat curve files found under %s.', base_dir);
end

%% -------------------- Load the original seed ("good") layout --------------------
good_layout_entry = struct('mode', {}, 'repeat_id', {}, 'seed', {}, ...
    'init_bestfun', {}, 'final_bestfun', {}, 'actual_iterations', {}, ...
    'stop_reason', {}, 'layout_12x12', {}, 'layout_row', {});
if isfile(good_mat)
    G = load(good_mat);
    if isfield(G, 'good_layout')
        if isfield(G, 'best_run_idx') && isfield(G, 'all_best_fitness') && ...
                G.best_run_idx >= 1 && G.best_run_idx <= numel(G.all_best_fitness)
            good_final = G.all_best_fitness(G.best_run_idx);
        else
            good_final = NaN;
        end
        good_row = double(G.good_layout(:)');
        gd = struct('mode', "good (seed)", 'repeat_id', NaN, 'seed', NaN, ...
            'init_bestfun', NaN, 'final_bestfun', good_final, ...
            'actual_iterations', NaN, 'stop_reason', "n/a", ...
            'layout_12x12', reshape(good_row, 12, 12), 'layout_row', good_row);
        good_layout_entry(end+1) = gd; %#ok<AGROW>
        fprintf('Loaded good/seed layout from %s (bestfun=%.2f)\n', good_mat, good_final);
    else
        fprintf('%s does not contain a variable named good_layout (skipping).\n', good_mat);
    end
else
    fprintf('No good_layout.mat found at %s (skipping seed-layout comparison).\n', good_mat);
end

%% -------------------- Save the layouts --------------------
save(out_mat, 'best_layouts', 'good_layout_entry');
fprintf('Saved %s\n', out_mat);

rows = table();
for di = 1:numel(good_layout_entry)
    d = good_layout_entry(di);
    rows = [rows; table(string(d.mode), d.repeat_id, d.seed, ...
        d.init_bestfun, d.final_bestfun, d.actual_iterations, string(d.stop_reason), ...
        'VariableNames', {'mode', 'repeat_id', 'seed', ...
        'init_bestfun', 'final_bestfun', 'actual_iterations', 'stop_reason'})]; %#ok<AGROW>
end
for di = 1:numel(best_layouts)
    d = best_layouts(di);
    rows = [rows; table(string(d.mode), d.repeat_id, d.seed, ...
        d.init_bestfun, d.final_bestfun, d.actual_iterations, string(d.stop_reason), ...
        'VariableNames', {'mode', 'repeat_id', 'seed', ...
        'init_bestfun', 'final_bestfun', 'actual_iterations', 'stop_reason'})]; %#ok<AGROW>
end
writetable(rows, out_summary);
fprintf('Saved %s\n', out_summary);

%% -------------------- Per-mode pixel-grid image --------------------
for di = 1:numel(best_layouts)
    d = best_layouts(di);
    fig = figure('Visible', 'off');
    imagesc(d.layout_12x12);
    colormap(gca, [1 1 1; 0 0 0]);   % 0 = white, 1 = black (metal pixel)
    clim([0 1]);
    axis image;
    set(gca, 'XTick', [], 'YTick', []);
    title(sprintf('Best layout (%s, rep=%d, bestfun=%.2f, iters=%d)', ...
        d.mode, d.repeat_id, d.final_bestfun, d.actual_iterations));
    out_png = fullfile(base_dir, d.mode, sprintf('best_layout_%s.png', d.mode));
    saveas(fig, out_png);
    close(fig);
    fprintf('Saved %s\n', out_png);
end

%% -------------------- Combined side-by-side comparison (one row) --------------------
% Good/seed layout goes first (reference), followed by each mode's best.
% No mixed-vs-warm panel comparison is drawn/labeled - just side-by-side.
panels = [good_layout_entry, best_layouts];
fig = figure('Visible', 'off', 'Position', [100 100 300 * numel(panels) + 100, 380]);
for di = 1:numel(panels)
    d = panels(di);
    subplot(1, numel(panels), di);
    imagesc(d.layout_12x12);
    colormap(gca, [1 1 1; 0 0 0]);
    clim([0 1]);
    axis image;
    set(gca, 'XTick', [], 'YTick', []);
    if isnan(d.repeat_id)
        title(sprintf('%s\nbestfun=%.2f', d.mode, d.final_bestfun));
    else
        title(sprintf('%s\nrep %d, bestfun=%.2f, iters=%d', ...
            d.mode, d.repeat_id, d.final_bestfun, d.actual_iterations));
    end
end
sgtitle('Best layout per mode (early stopping) vs. original good/seed layout');
saveas(fig, out_grid);
close(fig);
fprintf('Saved %s\n', out_grid);

%% -------------------- Hamming distance / similarity vs. good/seed ONLY --------------------
% Feed (port) pixels are forced to 1 in every layout, so they carry no
% design information; exclude them to avoid inflating the similarity score.
% Only mode-vs-good comparisons are computed - mixed vs. warm is not needed.
feed_idx = [6, 7];
n_free = 144 - numel(feed_idx);

hamming_rows = table();
if ~isempty(good_layout_entry)
    good_row = good_layout_entry(1).layout_row;
    good_row(feed_idx) = [];
    for di = 1:numel(best_layouts)
        d = best_layouts(di);
        row = d.layout_row;
        row(feed_idx) = [];
        hd = sum(row ~= good_row);
        pct = 100 * (1 - hd / n_free);
        hamming_rows = [hamming_rows; table(string(d.mode), d.repeat_id, hd, pct, ...
            'VariableNames', {'mode', 'repeat_id', 'hamming_distance_vs_good', ...
            'percent_identical_vs_good'})]; %#ok<AGROW>
    end
    writetable(hamming_rows, out_hamming);
    fprintf('Saved %s\n', out_hamming);

    fprintf('\n==== Similarity vs. good/seed layout (feed pixels excluded) ====\n');
    disp(hamming_rows);
else
    fprintf('No good/seed layout available - skipping Hamming distance comparison.\n');
end

fprintf(['\nDone. Load %s and index into ''best_layouts'' (per mode) or ' ...
    '''good_layout_entry'' (seed) to get layout_row (1x144) or layout_12x12.\n'], out_mat);
