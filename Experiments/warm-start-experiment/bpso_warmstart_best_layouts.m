%% BPSO warm-start experiment - retrieve best final layouts (post-processing only)
% Scans the per-repeat checkpoints produced by bpso_warmstart_experiment.m for
% each mode (random / mixed / warm), picks the repeat with the highest
% final_bestfun per mode, and pulls out its best_layout (the 12x12 pixel
% design, feed pixels already locked to 1).
%
% This does NOT re-run BPSO; it only reads curve_<mode>_rep*.mat files that
% already contain 'best_layout' (see bpso_warmstart_experiment.m).
%
% Also renders the original seed ("good") layout from
% bpso_grid_repeats_good_results/good_layout.mat, so it can be compared
% side-by-side against the best random/mixed/warm layouts.
%
% Outputs (in warmstart_results/):
%   best_layouts.mat          - struct array 'best_layouts', one entry per
%                               mode, with fields: mode, repeat_id, seed,
%                               init_bestfun, final_bestfun, layout_12x12,
%                               layout_row (1x144)
%   best_layouts_summary.csv  - one row per mode: which repeat won and its
%                               scores
%   best_layouts_grid.png     - side-by-side pixel-grid image of the best
%                               layout per mode, plus the good/seed layout
%   best_layouts_hamming.csv  - pairwise Hamming distance / percent-identical
%                               pixels between every pair of layouts (good +
%                               each mode's best), feed pixels excluded
% Outputs (in warmstart_results/<mode>/):
%   best_layout_<mode>.png    - pixel-grid image of that mode's best layout
% Outputs (in bpso_grid_repeats_good_results/):
%   good_layout.png           - pixel-grid image of the seed ("good") layout
%
% Usage:
%   run bpso_warmstart_best_layouts.m   (from anywhere; paths anchored to
%   the script folder). Run after bpso_warmstart_experiment.m has produced
%   results for the modes you care about.

clc;
close all;

%% -------------------- Settings --------------------
modes = {'random', 'mixed', 'warm'};   % modes to look for

%% -------------------- Path setup (anchored to this file) --------------------
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

base_dir = fullfile(script_dir, 'warmstart_results');
if ~exist(base_dir, 'dir')
    error('No warmstart_results folder at %s. Run bpso_warmstart_experiment.m first.', base_dir);
end

out_mat     = fullfile(base_dir, 'best_layouts.mat');
out_summary = fullfile(base_dir, 'best_layouts_summary.csv');
out_grid    = fullfile(base_dir, 'best_layouts_grid.png');
out_hamming = fullfile(base_dir, 'best_layouts_hamming.csv');

good_dir = fullfile(script_dir, 'bpso_grid_repeats_good_results');
good_mat = fullfile(good_dir, 'good_layout.mat');
out_good_png = fullfile(good_dir, 'good_layout.png');

%% -------------------- Find the winning repeat per mode --------------------
best_layouts = struct('mode', {}, 'repeat_id', {}, 'seed', {}, ...
    'init_bestfun', {}, 'final_bestfun', {}, 'layout_12x12', {}, 'layout_row', {});

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

    d.mode          = mode_str;
    d.repeat_id     = best_rid;
    d.seed          = best_S.seed;
    d.init_bestfun  = best_S.init_bestfun;
    d.final_bestfun = best_S.final_bestfun;
    d.layout_row    = double(best_S.best_layout(:)');
    d.layout_12x12  = reshape(d.layout_row, 12, 12);
    best_layouts(end+1) = d; %#ok<AGROW>

    fprintf('Mode "%s": best repeat %d (seed=%d, init=%.2f, final=%.2f)\n', ...
        mode_str, best_rid, d.seed, d.init_bestfun, d.final_bestfun);
end

if isempty(best_layouts)
    error('No per-repeat curve files found under %s.', base_dir);
end

%% -------------------- Load the original seed ("good") layout --------------------
good_layout_entry = struct('mode', {}, 'repeat_id', {}, 'seed', {}, ...
    'init_bestfun', {}, 'final_bestfun', {}, 'layout_12x12', {}, 'layout_row', {});
if isfile(good_mat)
    G = load(good_mat);
    if isfield(G, 'good_layout')
        gd.mode      = 'good (seed)';
        gd.repeat_id = NaN;
        gd.seed      = NaN;
        gd.init_bestfun = NaN;
        if isfield(G, 'best_run_idx') && isfield(G, 'all_best_fitness') && ...
                G.best_run_idx >= 1 && G.best_run_idx <= numel(G.all_best_fitness)
            gd.final_bestfun = G.all_best_fitness(G.best_run_idx);
        else
            gd.final_bestfun = NaN;
        end
        gd.layout_row   = double(G.good_layout(:)');
        gd.layout_12x12 = reshape(gd.layout_row, 12, 12);
        good_layout_entry(end+1) = gd; %#ok<AGROW>
        fprintf('Loaded good/seed layout from %s (bestfun=%.2f)\n', good_mat, gd.final_bestfun);

        % Standalone pixel-grid image, saved next to good_layout.mat.
        fig = figure('Visible', 'off');
        imagesc(gd.layout_12x12);
        colormap(gca, [1 1 1; 0 0 0]);
        clim([0 1]);
        axis image;
        set(gca, 'XTick', [], 'YTick', []);
        title(sprintf('Good/seed layout (bestfun=%.2f)', gd.final_bestfun));
        saveas(fig, out_good_png);
        close(fig);
        fprintf('Saved %s\n', out_good_png);
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
        d.init_bestfun, d.final_bestfun, ...
        'VariableNames', {'mode', 'repeat_id', 'seed', ...
        'init_bestfun', 'final_bestfun'})]; %#ok<AGROW>
end
for di = 1:numel(best_layouts)
    d = best_layouts(di);
    rows = [rows; table(string(d.mode), d.repeat_id, d.seed, ...
        d.init_bestfun, d.final_bestfun, ...
        'VariableNames', {'mode', 'repeat_id', 'seed', ...
        'init_bestfun', 'final_bestfun'})]; %#ok<AGROW>
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
    title(sprintf('Best layout (%s, rep=%d, bestfun=%.2f)', ...
        d.mode, d.repeat_id, d.final_bestfun));
    out_png = fullfile(base_dir, d.mode, sprintf('best_layout_%s.png', d.mode));
    saveas(fig, out_png);
    close(fig);
    fprintf('Saved %s\n', out_png);
end

%% -------------------- Combined side-by-side comparison --------------------
% Good/seed layout goes first (reference), followed by each mode's best.
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
        title(sprintf('%s\nrep %d, bestfun=%.2f', d.mode, d.repeat_id, d.final_bestfun));
    end
end
sgtitle('Best layout per mode vs. original good/seed layout');
saveas(fig, out_grid);
close(fig);
fprintf('Saved %s\n', out_grid);

%% -------------------- Pairwise Hamming distance / similarity --------------------
% Feed (port) pixels are forced to 1 in every layout, so they carry no
% design information; exclude them to avoid inflating the similarity score.
feed_idx = [6, 7];
n_free = 144 - numel(feed_idx);

n_panels = numel(panels);
panel_names = arrayfun(@(d) string(d.mode), panels);
hamming_dist = zeros(n_panels);        % number of differing free pixels
percent_identical = zeros(n_panels);   % 0-100 (100 = identical design)

for a = 1:n_panels
    row_a = panels(a).layout_row;
    row_a(feed_idx) = [];
    for b = 1:n_panels
        row_b = panels(b).layout_row;
        row_b(feed_idx) = [];
        hd = sum(row_a ~= row_b);
        hamming_dist(a, b) = hd;
        percent_identical(a, b) = 100 * (1 - hd / n_free);
    end
end

fprintf('\n==== Pairwise similarity (%% identical pixels, feed pixels excluded) ====\n');
Tsim = array2table(percent_identical, ...
    'VariableNames', matlab.lang.makeValidName(cellstr(panel_names)), ...
    'RowNames', cellstr(panel_names));
disp(Tsim);

% Long-format CSV: one row per ordered pair (easier to sort/filter than a matrix).
pairs = table();
for a = 1:n_panels
    for b = 1:n_panels
        if a == b
            continue;   % skip self-comparisons
        end
        pairs = [pairs; table(panel_names(a), panel_names(b), ...
            hamming_dist(a, b), percent_identical(a, b), ...
            'VariableNames', {'layout_a', 'layout_b', ...
            'hamming_distance', 'percent_identical'})]; %#ok<AGROW>
    end
end
writetable(pairs, out_hamming);
fprintf('Saved %s\n', out_hamming);

fprintf(['\nDone. Load %s and index into ''best_layouts'' (per mode) or ' ...
    '''good_layout_entry'' (seed) to get layout_row (1x144) or layout_12x12.\n'], out_mat);
