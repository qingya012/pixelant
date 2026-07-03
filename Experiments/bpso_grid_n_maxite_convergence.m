%% BPSO grid over n and maxite (simple driver)
% Same per-run output as inverse_design_using_bpso_with_TLfwdmodel.m.
% Loads the ONNX model once, loops over n x maxite, saves CSV + plots.
%
% Outputs (in bpso_grid_results/):
%   bpso_grid_results.csv
%   convergence_nXXX_maxiteYY.png
%   s11_nXXX_maxiteYY.png
%
% Usage:
%   cd to folder containing TLfwdmodel.onnx
%   bpso_grid_n_maxite

clc;
close all;

n_values = [200, 500, 1000];
maxite_values = [25, 50];
rng_seed = 42;              % set [] to leave RNG unchanged

results_dir = 'bpso_grid_results';
results_csv = fullfile(results_dir, 'bpso_grid_results.csv');

num_configs = numel(n_values) * numel(maxite_values);
fprintf('Grid: %d configs\n', num_configs);

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
num_pixels = 12;

pass_band = [center_fiu-4:center_fiu+4];
stop_band = [1:center_fiu-5, center_fiu+5:81];
pass_freq = freq(pass_band);
stop_freq = freq(stop_band);

wmax = 0.9;
wmin = 0.4;
c1 = 2;
c2 = 2;
maxrun = 1;

results = [];
run_id = 0;

for n = n_values
    for maxite = maxite_values
        run_id = run_id + 1;
        run_tag = sprintf('n%d_maxite%d', n, maxite);

        fprintf('\n========== Grid run %d/%d: n=%d, maxite=%d ==========\n', ...
            run_id, num_configs, n, maxite);

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
        [bestfun, bestrun] = max(fff)  % fitness is maximized (higher is better)
        best_variables = rgbest(bestrun, :)
        disp(sprintf('*********************************************************'));

        antenna_des = reshape(best_variables, 12, 12);
        output_new = predict(net, antenna_des);

        conv_path = fullfile(results_dir, ['convergence_' run_tag '.png']);
        s11_path = fullfile(results_dir, ['s11_' run_tag '.png']);

        fig_conv = figure('Visible', 'off');
        plot(ffmin(1:ffite(bestrun), bestrun), '-k');
        xlabel('Iteration');
        ylabel('Fitness function value');
        title(sprintf('PSO convergence (n=%d, maxite=%d)', n, maxite));
        grid on;
        saveas(fig_conv, conv_path);
        close(fig_conv);

        fig_s11 = figure('Visible', 'off');
        plot(freq, output_new(1, 1:81));
        legend('Reconstructed', 'Location', 'northeast');
        xlabel('freq');
        ylabel('Return Loss');
        title(sprintf('Predicted S11 (n=%d, maxite=%d, bestfun=%.2f)', n, maxite, bestfun));
        grid on;
        saveas(fig_s11, s11_path);
        close(fig_s11);

        des_ant = antenna_des;
        toc;

        results = [results; run_id, n, maxite, bestfun, bpso_elapsed_sec]; %#ok<AGROW>
        T = array2table(results, 'VariableNames', ...
            {'run_id', 'n', 'maxite', 'bestfun', 'bpso_elapsed_sec'});
        writetable(T, results_csv);

        fprintf('Saved %s\n', conv_path);
        fprintf('Saved %s\n', s11_path);
        fprintf('Saved row to %s\n', results_csv);
    end
end

fprintf('\nGrid complete.\n');
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
