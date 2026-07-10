%% BPSO inertia-weight sweep at README baseline (n=1000, maxite=50)
% Varies wmax/wmin only; c1, c2, and eval budget stay fixed.
%
% Outputs (in bpso_inertia_results/):
%   bpso_inertia_results.csv
%   convergence_wmax0p9_wmin0p4.png  (and one pair per config)
%   s11_wmax0p9_wmin0p4.png
%
% Usage:
%   cd to folder containing TLfwdmodel.onnx
%   bpso_inertia_weight_sweep

clc;
close all;

%% Baseline BPSO budget (README defaults)
n = 1000;
maxite = 50;
c1 = 2;
c2 = 2;
maxrun = 1;

%% Three inertia schedules: baseline, more exploitation, more exploration
weight_labels = {'default', 'exploit', 'explore'};
wmax_values = [0.9, 0.7, 0.95];
wmin_values = [0.4, 0.2, 0.6];

num_configs = numel(weight_labels);
rng_seed = 42;              % set [] to leave RNG unchanged

results_dir = 'bpso_inertia_results';
results_csv = fullfile(results_dir, 'bpso_inertia_results.csv');

fprintf('Inertia sweep: %d configs at n=%d, maxite=%d\n', num_configs, n, maxite);

if ~isfile('TLfwdmodel.onnx')
    error('TLfwdmodel.onnx not found in %s', pwd);
end

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

if ~isempty(rng_seed)
    rng(rng_seed);
end

fprintf('Loading TLfwdmodel.onnx ...\n');
net = importONNXNetwork('TLfwdmodel.onnx');

m = 144;
fmin = 1e9;
fmax = 5e9;
N = 81;
freq = linspace(fmin, fmax, N);
center_fiu = find(freq == 3.6e9);

pass_band = [center_fiu-4:center_fiu+4];
stop_band = [1:center_fiu-5, center_fiu+5:81];
pass_freq = freq(pass_band);
stop_freq = freq(stop_band);

T = table();

for cfg = 1:num_configs
    wmax = wmax_values(cfg);
    wmin = wmin_values(cfg);
    label = weight_labels{cfg};
    run_tag = sprintf('wmax%s_wmin%s', ...
        strrep(sprintf('%.2f', wmax), '.', 'p'), ...
        strrep(sprintf('%.2f', wmin), '.', 'p'));

    fprintf('\n========== Config %d/%d: %s (wmax=%.2f, wmin=%.2f) ==========\n', ...
        cfg, num_configs, label, wmax, wmin);

    tic;
    for run = 1:maxrun
        x = randi([0, 1], n, m);
        initial_inputmatrix = x;
        v = 0.1 * initial_inputmatrix;
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
        [fmin0, index0] = max(Error_vec);
        pbest = initial_inputmatrix;
        gbest = initial_inputmatrix(index0, :);

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
            ffmin(ite, run) = fmin;
            ffite(run) = ite;
            if fmin > fmin0
                gbest = pbest(index, :);
                fmin0 = fmin;
            end
            if ite == 1
                disp(sprintf('Iteration Best particle Objective fun'));
            end
            disp(sprintf('%8g %8g %8.4f', ite, index, fmin0));
            ite = ite + 1;
        end

        gbest
        antenna_de = reshape(gbest, 12, 12);
        antenna_de(6:7, 1) = 1;  % BUGFIX: force feed pixels (port line), matching in-loop scoring so bestfun is correct
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
        disp(sprintf('--------------------------------------'));
    end
    bpso_elapsed_sec = toc;

    disp(sprintf('\n'));
    disp(sprintf('*********************************************************'));
    disp(sprintf('Final Results-----------------------------'));
    [bestfun, bestrun] = max(fff)
    best_variables = rgbest(bestrun, :)
    best_variables([6, 7]) = 1;  % B-FIX: lock feed (port) pixels into the exported design vector so it is physically valid
    disp(sprintf('*********************************************************'));

    antenna_des = reshape(best_variables, 12, 12);
    antenna_des(6:7, 1) = 1;  % BUGFIX: force feed pixels (port line) so plotted S11 matches the optimized design
    output_new = predict(net, antenna_des);

    conv_path = fullfile(results_dir, ['convergence_' run_tag '.png']);
    s11_path = fullfile(results_dir, ['s11_' run_tag '.png']);

    fig_conv = figure('Visible', 'off');
    plot(ffmin(1:ffite(bestrun), bestrun), '-k');
    xlabel('Iteration');
    ylabel('Fitness function value');
    title(sprintf('PSO convergence (wmax=%.2f, wmin=%.2f, n=%d, maxite=%d)', ...
        wmax, wmin, n, maxite));
    grid on;
    saveas(fig_conv, conv_path);
    close(fig_conv);

    fig_s11 = figure('Visible', 'off');
    plot(freq, output_new(1, 1:81));
    legend('Reconstructed', 'Location', 'northeast');
    xlabel('freq');
    ylabel('Return Loss');
    title(sprintf('Predicted S11 (%s, bestfun=%.2f)', label, bestfun));
    grid on;
    saveas(fig_s11, s11_path);
    close(fig_s11);

    des_ant = antenna_des;
    toc;

    row = table(cfg, {label}, wmax, wmin, n, maxite, bestfun, bpso_elapsed_sec, ...
        'VariableNames', {'run_id', 'label', 'wmax', 'wmin', 'n', 'maxite', 'bestfun', 'bpso_elapsed_sec'});
    if height(T) == 0
        T = row;
    else
        T = [T; row]; %#ok<AGROW>
    end
    writetable(T, results_csv);

    fprintf('Saved %s\n', conv_path);
    fprintf('Saved %s\n', s11_path);
    fprintf('Saved row to %s\n', results_csv);
end

fprintf('\nInertia sweep complete.\n');
fprintf('  Results CSV: %s\n', results_csv);
fprintf('  Plots folder: %s\n', results_dir);
disp(T);

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
