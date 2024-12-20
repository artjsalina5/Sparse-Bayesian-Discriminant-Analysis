%% BCPD and DTW Implementation for Stress Detection
% This script implements Bayesian Change Point Detection (BCPD) and Dynamic Time Warping (DTW)
% to analyze physiological signal features and identify stress-related changes.

clc; clear;

%% Parameters
sampling_rate = 256; % Sampling frequency (Hz)
dtw_window_size = 60 * sampling_rate; % 60 seconds for DTW comparison
alpha_prior = 1; % Bayesian prior for mean
beta_prior = 1; % Bayesian prior for precision

%% Load Features
% Replace these placeholders with actual feature data
% `features` should be a struct with fields:
%   - time: Time stamps of the features
%   - data: Feature matrix (N x F), where N = samples, F = features
load('vital_features.mat'); % Output from SBDA or previous feature extraction

timestamps = vital_features.time;
features = vital_features.data;

%% Step 1: Bayesian Change Point Detection (BCPD)
% Detect change points in feature time series using BCPD

% Initialize variables for BCPD
[num_samples, num_features] = size(features);
change_points = [];
segment_means = [];
segment_vars = [];

% Iterate through features to identify change points
for i = 2:num_samples
    % Compute posterior parameters
    segment = features(1:i, :); % Current segment
    segment_mean = mean(segment, 1);
    segment_var = var(segment, 0, 1);

    alpha_post = alpha_prior + size(segment, 1) / 2;
    beta_post = beta_prior + 0.5 * sum((segment - segment_mean).^2, 'all');

    % Posterior mean and variance
    post_mean = beta_post / alpha_post;
    post_var = 1 / alpha_post;

    % Detect significant changes
    if ~isempty(segment_means)
        if any(abs(segment_mean - segment_means(end, :)) > 2 * sqrt(segment_vars(end, :))) || ...
           any(abs(segment_var - segment_vars(end, :)) > 0.5 * segment_vars(end, :))
            change_points = [change_points; i];
        end
    end

    % Update segment statistics
    segment_means = [segment_means; segment_mean];
    segment_vars = [segment_vars; segment_var];
end

fprintf('Detected %d change points using BCPD.\n', length(change_points));

%% Step 2: Dynamic Time Warping (DTW) for Validation
% Compare feature segments using DTW to validate BCPD change points

dtw_distances = zeros(length(change_points) - 1, 1);

for i = 1:length(change_points) - 1
    % Define segments around change points
    start_idx = change_points(i);
    end_idx = min(change_points(i+1), num_samples);
    segment1 = features(start_idx:end_idx, :);

    if i + 2 <= length(change_points)
        next_start_idx = change_points(i+1);
        next_end_idx = min(change_points(i+2), num_samples);
        segment2 = features(next_start_idx:next_end_idx, :);
    else
        segment2 = features(end_idx:end, :);
    end

    % Compute DTW distance
    dtw_distances(i) = compute_dtw(segment1, segment2);
end

fprintf('Computed DTW distances for %d segment pairs.\n', length(dtw_distances));

%% Step 3: Visualize Results
% Plot BCPD results and DTW distances
figure;
subplot(2, 1, 1);
plot(timestamps, features(:, 1), 'b', 'LineWidth', 1.5);
hold on;
for cp = change_points'
    xline(timestamps(cp), 'r--', 'LineWidth', 1.5);
end
title('Feature Time Series with Detected Change Points');
xlabel('Time (s)');
ylabel('Feature Value');
hold off;

subplot(2, 1, 2);
bar(dtw_distances);
title('DTW Distances Between Segments');
xlabel('Segment Pair Index');
ylabel('DTW Distance');

%% Supporting Functions

function distance = compute_dtw(features1, features2)
    % Compute DTW distance between two feature sets.
    %
    % Inputs:
    % - features1: Feature matrix for the first segment (N1 x F)
    % - features2: Feature matrix for the second segment (N2 x F)
    %
    % Output:
    % - distance: DTW distance between the two feature sets

    [distance, ~] = dtw(features1, features2);
end