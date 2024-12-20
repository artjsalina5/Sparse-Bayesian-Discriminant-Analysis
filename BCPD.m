function change_points = BayesianCPD_withPlots(features, window_size, step_size, timestamps)
    % BayesianCPD_withPlots: Detect change points and plot results
    %
    % Inputs:
    % - features: A matrix where rows are time points and columns are features
    % - window_size: Size of each window (in time points)
    % - step_size: Step size for sliding the window (in time points)
    % - timestamps: Time vector for plotting
    %
    % Outputs:
    % - change_points: Indices of detected change points

    % Parameters
    alpha_prior = 1; % Prior for the mean
    beta_prior = 1;  % Prior for variance
    [n, p] = size(features); % n = time points, p = number of features

    % Initialize outputs
    change_points = [];
    mean_segments = [];
    var_segments = [];
    
    % Iterate through the signal using a sliding window
    for start_idx = 1:step_size:(n - window_size + 1)
        % Extract the current window
        end_idx = start_idx + window_size - 1;
        current_window = features(start_idx:end_idx, :);

        % Compute statistics for the current window
        mean_window = mean(current_window, 1); % Mean of each feature
        var_window = var(current_window, 0, 1); % Variance of each feature

        % Bayesian update of posterior parameters
        alpha_post = alpha_prior + size(current_window, 1) / 2;
        beta_post = beta_prior + 0.5 * sum((current_window - mean_window).^2, 'all');

        % Change point detection based on significant mean or variance shift
        if ~isempty(mean_segments)
            prev_mean = mean_segments(end, :);
            prev_var = var_segments(end, :);

            % Check for significant deviation in mean or variance
            if any(abs(mean_window - prev_mean) > 2 * sqrt(var_window)) || ...
               any(abs(var_window - prev_var) > 0.5 * prev_var)
                change_points = [change_points; start_idx];
            end
        end

        % Update segment statistics
        mean_segments = [mean_segments; mean_window];
        var_segments = [var_segments; var_window];
    end

    % Plotting
    figure;
    for i = 1:p
        subplot(p, 1, i);
        plot(timestamps, features(:, i));
        hold on;
        % Mark change points
        for cp = change_points'
            xline(timestamps(cp), 'r--', 'LineWidth', 1.5);
        end
        hold off;
        title(['Feature ', num2str(i)]);
        xlabel('Time');
        ylabel('Value');
    end
    
    % Output detected change points
    fprintf('Detected %d change points.\n', length(change_points));
end
