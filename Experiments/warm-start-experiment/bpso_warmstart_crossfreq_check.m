%% Cross-frequency sanity check (compute-only, no BPSO, no modes)
% Question: the "good" layout was found by BPSO targeting 3.6 GHz. Is it
% ALSO decent at a different target frequency (e.g. 2 GHz), just by luck?
%
% This does NOT run any optimization. good_layout is a fixed 12x12 design;
% the forward model always returns its response across the FULL 1-5 GHz
% sweep in one predict() call, so checking a different frequency is just
% reading a different index out of an output you already have.
%
% Outputs (in warmstart_results/):
%   crossfreq_check_s11.png   - full S11 curve, both target freqs marked
%   crossfreq_check.csv       - S11 value at each target freq + verdict
%
% Usage:
%   Edit target_freqs_ghz below if you want to check other frequencies,
%   then run this script (from anywhere; paths anchored to this file).

clc;
close all;

%% -------------------- Settings --------------------
target_freqs_ghz = [3.6, 2.0];   % [reference freq the layout was optimized for, new freq to check]
half_width_bins  = 4;            % same passband half-width used during BPSO (+/-4 bins)

%% -------------------- Path setup (anchored to this file) --------------------
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end

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
        'root (%s).'], fullfile(script_dir, '..', '..'));
end

good_mat = fullfile(script_dir, 'bpso_grid_repeats_good_results', 'good_layout.mat');
if ~isfile(good_mat)
    error('good_layout.mat not found at %s.', good_mat);
end

results_dir = fullfile(script_dir, 'warmstart_results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
out_png = fullfile(results_dir, 'crossfreq_check_s11.png');
out_csv = fullfile(results_dir, 'crossfreq_check.csv');

%% -------------------- Load layout + model --------------------
S = load(good_mat, 'good_layout');
if ~isfield(S, 'good_layout')
    error('%s does not contain a variable named good_layout.', good_mat);
end
layout = reshape(double(S.good_layout), 12, 12);
layout(6:7, 1) = 1;   % force feed/port pixels, same convention as every other script

fprintf('Loading %s ...\n', onnx_path);
net = importONNXNetwork(onnx_path);

fmin_hz = 1e9;
fmax_hz = 5e9;
N = 81;
freq = linspace(fmin_hz, fmax_hz, N);

%% -------------------- One forward pass: full S11 sweep --------------------
output = predict(net, layout);
s11 = output(1, 1:81);

%% -------------------- Score each target frequency --------------------
rows = table();
fig = figure('Visible', 'off', 'Position', [100 100 760 480]);
plot(freq, s11, '-k', 'LineWidth', 1.3);
hold on;
colors = {[0.85 0.33 0.10], [0 0.45 0.74], [0.47 0.67 0.19], [0.49 0.18 0.56]};

for fi = 1:numel(target_freqs_ghz)
    f_hz = target_freqs_ghz(fi) * 1e9;
    idx = find(abs(freq - f_hz) < 1);   % tolerant match onto the 50 MHz grid
    if isempty(idx)
        warning('%.3f GHz is not on the frequency grid (skipping).', target_freqs_ghz(fi));
        continue;
    end
    idx = idx(1);
    band = max(1, idx - half_width_bins) : min(81, idx + half_width_bins);
    val_at_freq  = s11(idx);
    val_band_avg = mean(s11(band));

    plot(freq(idx), val_at_freq, 'o', 'MarkerSize', 8, 'MarkerFaceColor', colors{mod(fi-1,4)+1}, ...
        'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    text(freq(idx), val_at_freq, sprintf('  %.1f GHz', target_freqs_ghz(fi)), ...
        'FontSize', 10, 'Color', colors{mod(fi-1,4)+1});

    rows = [rows; table(target_freqs_ghz(fi), idx, val_at_freq, val_band_avg, ...
        'VariableNames', {'target_freq_ghz', 'freq_index', 's11_at_freq', 's11_band_avg'})]; %#ok<AGROW>

    fprintf('%.1f GHz (index %d): s11_at_freq=%.3f  band_avg(+/-%d bins)=%.3f\n', ...
        target_freqs_ghz(fi), idx, val_at_freq, half_width_bins, val_band_avg);
end
hold off;
grid on;
xlabel('freq');
ylabel('Return Loss');
title(sprintf('Cross-frequency check: good/seed layout (optimized for %.1f GHz)', target_freqs_ghz(1)));
saveas(fig, out_png);
close(fig);
fprintf('Saved %s\n', out_png);

writetable(rows, out_csv);
fprintf('Saved %s\n', out_csv);

fprintf(['\nInterpretation: this layout was optimized to be HIGH near the first ' ...
    'listed frequency and LOW elsewhere. If s11_band_avg at the other frequency ' ...
    'is also high, the design happens to work there "for free"; if it is low, ' ...
    'this layout gives no head start for that target (as expected, but worth' ...
    ' confirming before running a warm-start BPSO search targeting it).\n']);
