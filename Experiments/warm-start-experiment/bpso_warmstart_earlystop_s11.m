%% BPSO warm-start + early-stopping experiment - S11 regeneration (post-processing only)
% Regenerates the S11 (return-loss) curves for the early-stopping experiment
% runs ("mixed" and "warm" only - "random" is not run with early stopping,
% see bpso_warmstart_early_stopping.m for why).
%
% This does NOT re-run BPSO. Each repeat checkpoint produced by
% bpso_warmstart_early_stopping.m stored the winning design (best_layout,
% with feed pixels already locked) at whatever iteration it actually stopped,
% so the S11 achieved by that run can be reproduced EXACTLY by a single
% forward pass through TLfwdmodel.onnx - no digitising or approximation needed.
%
% Outputs (in warmstart_earlystop_results/<mode>/):
%   s11_<mode>_rep<k>.png              - one S11 plot per repeat (title
%                                         includes actual_iterations/stop_reason)
% Outputs (in warmstart_earlystop_results/):
%   earlystop_s11_best_overlay.png     - best design per mode, overlaid
%   earlystop_s11_index.csv            - which repeat/design each curve came
%                                         from, plus actual_iterations/stop_reason
%
% Usage:
%   run bpso_warmstart_earlystop_s11.m   (from anywhere; paths anchored to
%   the script folder). Run after bpso_warmstart_early_stopping.m has
%   produced results for the modes you care about.

clc;
close all;

%% -------------------- Settings --------------------
modes       = {'mixed', 'warm'};
mode_colors = {[0 0.45 0.74], [0.85 0.33 0.10]};  % blue, orange (matches bpso_warmstart_earlystop_plots.m)

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

base_dir = fullfile(script_dir, 'warmstart_earlystop_results');
if ~exist(base_dir, 'dir')
    error(['No warmstart_earlystop_results folder at %s. ' ...
        'Run bpso_warmstart_early_stopping.m first.'], base_dir);
end

out_overlay = fullfile(base_dir, 'earlystop_s11_best_overlay.png');
out_index   = fullfile(base_dir, 'earlystop_s11_index.csv');

%% -------------------- Problem setup (same as experiment) --------------------
fprintf('Loading %s ...\n', onnx_path);
net = importONNXNetwork(onnx_path);

fmin_hz = 1e9;
fmax_hz = 5e9;
N = 81;
freq = linspace(fmin_hz, fmax_hz, N);
center_fiu = find(freq == 3.6e9);
pass_band = [center_fiu-4:center_fiu+4];   % 3.6 GHz +/- 4 bins (marked on overlay)

%% -------------------- Regenerate S11 per repeat --------------------
best_per_mode = struct('mode', {}, 'color', {}, 's11', {}, ...
    'repeat_id', {}, 'final_bestfun', {}, 'actual_iterations', {});
index_rows = table();

for mi = 1:numel(modes)
    mode_str = modes{mi};
    mdir = fullfile(base_dir, mode_str);
    files = dir(fullfile(mdir, sprintf('curve_%s_rep*.mat', mode_str)));
    if isempty(files)
        fprintf('No data for mode "%s" (skipping).\n', mode_str);
        continue;
    end

    best_val = -inf;
    best_s11 = [];
    best_rep = NaN;
    best_ai  = NaN;

    for k = 1:numel(files)
        fpath = fullfile(files(k).folder, files(k).name);
        S = load(fpath);
        tok = regexp(files(k).name, 'rep(\d+)\.mat$', 'tokens', 'once');
        rid = str2double(tok{1});

        % Reproduce the exact S11 the run achieved: reshape the saved design,
        % re-assert the feed (port) pixels, and run one forward pass.
        antenna = reshape(double(S.best_layout), 12, 12);
        antenna(6:7, 1) = 1;
        output = predict(net, antenna);
        s11 = output(1, 1:81);

        if isfield(S, 'final_bestfun'), fval = S.final_bestfun; else, fval = NaN; end
        if isfield(S, 'actual_iterations'), ai = S.actual_iterations; else, ai = NaN; end
        if isfield(S, 'stop_reason'), sr = char(S.stop_reason); else, sr = 'unknown'; end
        if isfield(S, 'maxite'), mx = S.maxite; else, mx = NaN; end

        % Per-repeat S11 plot.
        s11_path = fullfile(mdir, sprintf('s11_%s_rep%d.png', mode_str, rid));
        fig = figure('Visible', 'off');
        plot(freq, s11);
        legend('Reconstructed', 'Location', 'northeast');
        xlabel('freq');
        ylabel('Return Loss');
        title(sprintf('Predicted S11 (%s, rep=%d, bestfun=%.2f, iters=%d/%d, stop=%s)', ...
            mode_str, rid, fval, ai, mx, sr), 'Interpreter', 'none');
        grid on;
        saveas(fig, s11_path);
        close(fig);
        fprintf('Saved %s\n', s11_path);

        index_rows = [index_rows; table(string(mode_str), rid, fval, ai, string(sr), ...
            string(s11_path), 'VariableNames', ...
            {'mode', 'repeat_id', 'final_bestfun', 'actual_iterations', 'stop_reason', 's11_png'})]; %#ok<AGROW>

        if fval > best_val
            best_val = fval;
            best_s11 = s11;
            best_rep = rid;
            best_ai  = ai;
        end
    end

    d.mode = mode_str;
    d.color = mode_colors{mi};
    d.s11 = best_s11;
    d.repeat_id = best_rep;
    d.final_bestfun = best_val;
    d.actual_iterations = best_ai;
    best_per_mode(end+1) = d; %#ok<AGROW>
    fprintf('Mode "%s": best repeat %d (bestfun=%.2f, iters=%d)\n', ...
        mode_str, best_rep, best_val, best_ai);
end

if isempty(best_per_mode)
    error('No per-repeat curve files found under %s.', base_dir);
end

%% -------------------- Overlay: best design per mode --------------------
fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
hold on;
legend_entries = {};
for di = 1:numel(best_per_mode)
    d = best_per_mode(di);
    plot(freq, d.s11, '-', 'Color', d.color, 'LineWidth', 1.8);
    legend_entries{end+1} = sprintf('%s (rep %d, bestfun=%.1f, iters=%d)', ...
        d.mode, d.repeat_id, d.final_bestfun, d.actual_iterations); %#ok<AGROW>
end
% Shade the passband (3.6 GHz +/- 4 bins) for context.
yl = ylim;
xp = [freq(pass_band(1)), freq(pass_band(end))];
patch([xp(1) xp(2) xp(2) xp(1)], [yl(1) yl(1) yl(2) yl(2)], ...
    [0.9 0.9 0.6], 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');
ylim(yl);
hold off;
grid on;
xlabel('freq');
ylabel('Return Loss');
title('Warm-start + early stopping: best S11 per mode');
legend(legend_entries, 'Location', 'northeast');
saveas(fig, out_overlay);
close(fig);
fprintf('Saved %s\n', out_overlay);

%% -------------------- Index CSV --------------------
if ~isempty(index_rows)
    index_rows = sortrows(index_rows, {'mode', 'repeat_id'});
    writetable(index_rows, out_index);
    fprintf('Saved %s\n', out_index);
end

fprintf(['\nDone. Per-repeat S11 in warmstart_earlystop_results/<mode>/, ' ...
    'overlay + index at warmstart_earlystop_results/ top level.\n']);
