%% BPSO baseline repeats with per-run layout capture and good_layout.mat export
% Same BPSO logic and baseline parameters as inverse_design_using_bpso_with_TLfwdmodel.m.
% Runs the README baseline (n=1000, maxite=50) num_repeats times with independent
% random initializations, then saves the overall best layout for warm-start experiments.
%
% Outputs (in bpso_grid_repeats_results/):
%   bpso_grid_results.csv  - one row per repeat (raw runs)
%   bpso_grid_summary.csv  - mean/std bestfun and time per (n, maxite)
%   convergence_nXXX_maxiteYY_repZZ.png
%   s11_nXXX_maxiteYY_repZZ.png
%   good_layout.mat        - overall best layout across all repeats
%
% Usage:
%   cd to folder containing TLfwdmodel.onnx
%   bpso_baseline_repeats_good_layout

clc;
close all;

% Baseline BPSO settings (README defaults; do not change for this experiment)
n_values = 1000;
maxite_values = 50;
num_repeats = 10;           % 10 independent random-initialization runs
rng_seed = 42;              % base seed; each repeat uses rng_seed + config_id*100 + repeat_id

results_dir = 'bpso_grid_repeats_results';
raw_results_csv = fullfile(results_dir, 'bpso_grid_results.csv');
summary_csv = fullfile(results_dir, 'bpso_grid_summary.csv');
good_layout_mat = fullfile(results_dir, 'good_layout.mat');

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

num_configs = numel(n_values) * numel(maxite_values);
fprintf('Baseline BPSO: %d configs x %d repeats = %d total runs\n', ...
    num_configs, num_repeats, num_configs * num_repeats);

if ~isfile('TLfwdmodel.onnx')
    error('TLfwdmodel.onnx not found in %s', pwd);
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

raw_results = [];
summary_results = [];
run_id = 0;
config_id = 0;

for n = n_values
    for maxite = maxite_values
        config_id = config_id + 1;
        rep_bestfun = nan(num_repeats, 1);
        rep_time = nan(num_repeats, 1);

        % Per-repeat best layouts and fitness values for warm-start export
        all_best_layouts = nan(num_repeats, m);
        all_best_fitness = nan(num_repeats, 1);

        for repeat_id = 1:num_repeats
            run_idx = repeat_id;
            run_id = run_id + 1;
            run_tag = sprintf('n%d_maxite%d_rep%d', n, maxite, repeat_id);
            if ~isempty(rng_seed)
                rng(rng_seed + config_id * 100 + repeat_id);
            end

            fprintf('\n========== Config %d/%d (n=%d, maxite=%d), repeat %d/%d (run %d) ==========\n', ...
                config_id, num_configs, n, maxite, repeat_id, num_repeats, run_id);

            tic;
            for run = 1:maxrun
                x = randi([0, 1], n, m);
                initial_inputmatrix = x;
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
                [fmin0, index0] = max(Error_vec(:,1));
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
            [bestfun, bestrun] = max(fff)  % fitness is maximized (higher is better)
            best_variables = rgbest(bestrun, :)
            best_variables([6, 7]) = 1;  % B-FIX: lock feed (port) pixels into the exported design vector so it is physically valid
            disp(sprintf('*********************************************************'));

            % Store this repeat's best layout and fitness for warm-start export
            all_best_layouts(run_idx, :) = best_variables;
            all_best_fitness(run_idx) = bestfun;

            antenna_des = reshape(best_variables, 12, 12);
            antenna_des(6:7, 1) = 1;  % BUGFIX: force feed pixels (port line) so plotted S11 matches the optimized design
            output_new = predict(net, antenna_des);

            conv_path = fullfile(results_dir, ['convergence_' run_tag '.png']);
            s11_path = fullfile(results_dir, ['s11_' run_tag '.png']);

            fig_conv = figure('Visible', 'off');
            plot(ffmin(1:ffite(bestrun), bestrun), '-k');
            xlabel('Iteration');
            ylabel('Fitness function value');
            title(sprintf('PSO convergence (n=%d, maxite=%d, rep=%d)', n, maxite, repeat_id));
            grid on;
            saveas(fig_conv, conv_path);
            close(fig_conv);

            fig_s11 = figure('Visible', 'off');
            plot(freq, output_new(1, 1:81));
            legend('Reconstructed', 'Location', 'northeast');
            xlabel('freq');
            ylabel('Return Loss');
            title(sprintf('Predicted S11 (n=%d, maxite=%d, rep=%d, bestfun=%.2f)', ...
                n, maxite, repeat_id, bestfun));
            grid on;
            saveas(fig_s11, s11_path);
            close(fig_s11);

            des_ant = antenna_des;
            toc;
            fprintf('Saved plots to %s and %s\n', conv_path, s11_path);

            rep_bestfun(repeat_id) = bestfun;
            rep_time(repeat_id) = bpso_elapsed_sec;

            raw_results = [raw_results; ...
                run_id, config_id, repeat_id, n, maxite, bestfun, bpso_elapsed_sec]; %#ok<AGROW>
            Traw = array2table(raw_results, 'VariableNames', ...
                {'run_id', 'config_id', 'repeat_id', 'n', 'maxite', 'bestfun', 'bpso_elapsed_sec'});
            writetable(Traw, raw_results_csv);
            fprintf('Saved raw row to %s\n', raw_results_csv);
        end

        % Overall best layout across all baseline repeats
        [~, best_run_idx] = max(all_best_fitness);
        good_layout = all_best_layouts(best_run_idx, :);
        save(good_layout_mat, 'good_layout', 'best_run_idx', 'all_best_fitness');
        fprintf('\nSaved overall best layout (repeat %d, bestfun=%.4f) to %s\n', ...
            best_run_idx, all_best_fitness(best_run_idx), good_layout_mat);

        mean_bestfun = mean(rep_bestfun);
        std_bestfun = std(rep_bestfun);
        mean_time = mean(rep_time);
        std_time = std(rep_time);

        summary_results = [summary_results; ...
            config_id, n, maxite, num_repeats, ...
            mean_bestfun, std_bestfun, mean_time, std_time]; %#ok<AGROW>
        Tsum = array2table(summary_results, 'VariableNames', ...
            {'config_id', 'n', 'maxite', 'num_repeats', ...
            'mean_bestfun', 'std_bestfun', 'mean_bpso_elapsed_sec', 'std_bpso_elapsed_sec'});
        writetable(Tsum, summary_csv);

        fprintf('\n--- Summary for n=%d, maxite=%d (%d repeats) ---\n', n, maxite, num_repeats);
        fprintf('  mean(bestfun) = %.4f,  std(bestfun) = %.4f\n', mean_bestfun, std_bestfun);
        fprintf('  mean(time)    = %.2f s, std(time)    = %.2f s\n', mean_time, std_time);
        fprintf('  Updated %s\n', summary_csv);
    end
end

fprintf('\nBaseline repeat study complete.\n');
fprintf('  Results folder: %s\n', results_dir);
fprintf('  Raw runs:  %s (%d rows)\n', raw_results_csv, height(Traw));
fprintf('  Summary:   %s (%d configs)\n', summary_csv, height(Tsum));
fprintf('  Warm-start: %s\n', good_layout_mat);
disp(Tsum);

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
