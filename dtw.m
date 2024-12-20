%% Implementation of DTW and BCPD for SBDA Features
% This script takes vital features (output of SBDA), computes DTW, and
% applies Bayesian Change Point Detection (BCPD) to identify state transitions.

clc; clear;

%% Parameters
sampling_rate = 256; % Sampling frequency in Hz
window_size = 10 * sampling_rate; % 10-second windows
step_size = 5 * sampling_rate; % 50% overlap
alpha_prior = 1; % Bayesian prior for mean
beta_prior = 1; % Bayesian prior for precision

%% Load Features (Output of SBDA)
% Assume we have a struct `vital_features` containing the SBDA output
% vital_features.time: Timestamps of the features
% vital_features.data: Matrix (N x F) of extracted features, where
% N = number of samples and F = number of features
load('vital_features.mat'); % Replace with actual SBDA output file

% Extract data and timestamps
timestamps = vital_features.time; % Timestamps (seconds)
features = vital_features.data; % Extracted features (N x F)

%% Step 1: Segment Features for DTW
% Divide features into overlapping windows
[feature_segments, num_segments] = segment_features(features, window_size, step_size);

% Initialize DTW distance matrix
dtw_distances = zeros(num_segments, num_segments);

%% Step 2: Compute DTW for Feature Alignment
% Compare feature windows pairwise using DTW
for i = 1:num_segments
    for j = i:num_segments
        dtw_distances(i, j) = compute_dtw(feature_segments{i}, feature_segments{j});
        dtw_distances(j, i) = dtw_distances(i, j); % Symmetric matrix
    end
end

%% Step 3: Apply Bayesian Change Point Detection (BCPD)
% Initialize variables for BCPD
mean_dtw = mean(dtw_distances, 2); % Mean DTW distance for each segment
var_dtw = var(dtw_distances, 0, 2); % Variance of DTW distances
change_points = []; % Store detected change points

% Iterate through DTW distance matrix
for i = 2:num_segments
    % Update posterior parameters
    alpha_post = alpha_prior + i / 2;
    beta_post = beta_prior + 0.5 * sum((mean_dtw(1:i) - mean(mean_dtw(1:i))).^2);

    % Posterior mean and variance
    post_mean = beta_post / alpha_post; % Posterior mean
    post_var = 1 / alpha_post; % Posterior variance

    % Detect change points based on posterior shifts
    if abs(mean_dtw(i) - post_mean) > 2 * sqrt(post_var) || ...
       abs(var_dtw(i) - post_var) > 0.5 * post_var
        change_points = [change_points; i];
    end
end

fprintf('Detected %d change points.\n', length(change_points));

%% Step 4: Visualize Results
% Plot DTW distance matrix
figure;
imagesc(dtw_distances);
colorbar;
title('DTW Distance Matrix');
xlabel('Segment Index');
ylabel('Segment Index');

% Plot mean DTW distances with detected change points
figure;
plot(1:num_segments, mean_dtw, '-b', 'LineWidth', 1.5);
hold on;
for cp = change_points'
    xline(cp, 'r--', 'LineWidth', 1.5);
end
title('Mean DTW Distances with Detected Change Points');
xlabel('Segment Index');
ylabel('Mean DTW Distance');
hold off;

%% Supporting Functions

function [segments, num_segments] = segment_features(features, window_size, step_size)
    % Divide features into overlapping windows.
    %
    % Inputs:
    % - features: Feature matrix (N x F)
    % - window_size: Size of each window (in samples)
    % - step_size: Step size for overlapping windows (in samples)
    %
    % Outputs:
    % - segments: Cell array of feature windows
    % - num_segments: Total number of segments

    [num_samples, ~] = size(features);
    num_segments = floor((num_samples - window_size) / step_size) + 1;
    segments = cell(num_segments, 1);

    for i = 1:num_segments
        start_idx = (i-1) * step_size + 1;
        end_idx = start_idx + window_size - 1;
        segments{i} = features(start_idx:end_idx, :);
    end
end

function distance = compute_dtw(features1, features2)
    % Compute DTW distance between two feature sets.
    %
    % Inputs:
    % - features1: Feature matrix for the first segment (W1 x F)
    % - features2: Feature matrix for the second segment (W2 x F)
    %
    % Output:
    % - distance: DTW distance between the two feature sets

    [distance, ~] = dtw(features1, features2);
end