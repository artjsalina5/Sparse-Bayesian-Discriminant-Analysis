%% Feature Extraction from ECG and BVP Signals at 256 Hz
% This script processes ECG and BVP signals to extract relevant features
% for further analysis, such as Bayesian Change Point Detection (BCPD).

%% Load and Simulate Signals
% Replace the simulated signals below with actual preprocessed data.
clc; clear;

% Simulated ECG and BVP signals
ecg_signal = randn(1, 256 * 60); % Example ECG signal (60 seconds at 256 Hz)
bvp_signal = rand(1, 256 * 60);  % Example BVP signal (60 seconds at 256 Hz)
sampling_rate = 256;             % Sampling rate (Hz)

%% ECG Signal Processing
% Step 1: Clean ECG Signal (Bandpass Filter)
% Bandpass filter the ECG signal between 0.5 and 40 Hz to remove noise
ecg_cleaned = bandpass(ecg_signal, [0.5, 40], sampling_rate);

% Step 2: Detect R-peaks using Pan-Tompkins Algorithm
% Implement the Pan-Tompkins algorithm to detect R-peaks in the ECG signal
[r_peaks, ecg_filtered] = ecg_findpeaks(ecg_cleaned, sampling_rate);

% Step 3: Compute NN intervals (Inter-beat Intervals)
% Calculate the time intervals between successive R-peaks (NN intervals)
nn_intervals = diff(r_peaks) / sampling_rate * 1000; % Convert to ms

% Step 4: Calculate HRV-related features
% Compute various Heart Rate Variability (HRV) metrics
max_hrv = std(nn_intervals);                 % Max HRV (SDNN)
mean_hr = 60000 / mean(nn_intervals);        % Mean HR (bpm)
rmssd = sqrt(mean(diff(nn_intervals).^2));   % RMSSD
entropy = approximate_entropy(nn_intervals); % Approximate Entropy
hf_power = bandpower(nn_intervals, sampling_rate, [0.15, 0.4]); % HF power (0.15-0.4 Hz)

%% BVP Signal Processing
% Step 1: Clean BVP Signal (Bandpass Filter)
% Bandpass filter the BVP signal between 0.5 and 8 Hz to remove noise
bvp_cleaned = bandpass(bvp_signal, [0.5, 8], sampling_rate);

% Step 2: Detect Peaks in BVP Signal
% Detect systolic peaks in the BVP signal, corresponding to heartbeats
[bvp_peaks, bvp_filtered] = bvp_findpeaks(bvp_cleaned, sampling_rate);

% Step 3: Compute Inter-Beat Intervals (IBIs) from BVP
% Calculate the time intervals between successive BVP peaks (IBIs)
ibi_intervals = diff(bvp_peaks) / sampling_rate * 1000; % Convert to ms

% Step 4: Calculate PRV-related features
% Compute various Pulse Rate Variability (PRV) metrics
max_prv = std(ibi_intervals);                 % Max PRV (SDNN)
mean_pr = 60000 / mean(ibi_intervals);        % Mean Pulse Rate (bpm)
rmssd_prv = sqrt(mean(diff(ibi_intervals).^2)); % RMSSD
entropy_prv = approximate_entropy(ibi_intervals); % Approximate Entropy
hf_power_prv = bandpower(ibi_intervals, sampling_rate, [0.15, 0.4]); % HF power (0.15-0.4 Hz)

%% Combine and Display Features
% Store features in a struct for further analysis
features = struct();
% ECG Features
features.Max_HRV = max_hrv;
features.Mean_HR = mean_hr;
features.RMSSD = rmssd;
features.Entropy_HRV = entropy;
features.HF_Power_HRV = hf_power;
% BVP Features
features.Max_PRV = max_prv;
features.Mean_PR = mean_pr;
features.RMSSD_PRV = rmssd_prv;
features.Entropy_PRV = entropy_prv;
features.HF_Power_PRV = hf_power_prv;

% Display extracted features
disp('Extracted Features:');
disp(features);

%% Save Features for Further Analysis
% Save features to a CSV file for BCPD or other analyses
features_table = struct2table(features);
writetable(features_table, 'ecg_bvp_features.csv');
disp('Features saved to ecg_bvp_features.csv.');

%% Supporting Functions
% Functions for peak detection, entropy calculation, etc.

function [r_peaks, ecg_filtered] = ecg_findpeaks(ecg_signal, sampling_rate)
    % Find R-peaks in the ECG signal using Pan-Tompkins algorithm
    % Inputs:
    % - ecg_signal: Cleaned ECG signal
    % - sampling_rate: Sampling rate (Hz)
    % Outputs:
    % - r_peaks: Indices of detected R-peaks
    % - ecg_filtered: Filtered ECG signal

    % Bandpass filter the ECG signal between 0.5 and 40 Hz
    ecg_filtered = bandpass(ecg_signal, [0.5, 40], sampling_rate);

    % Square the filtered signal to enhance R-peaks
    squared_signal = ecg_filtered.^2;

    % Apply a moving average filter to smooth the signal
    avg_signal = movmean(squared_signal, round(0.12 * sampling_rate));

    % Detect peaks in the smoothed signal corresponding to R-peaks
    [~, r_peaks] = findpeaks(avg_signal, 'MinPeakHeight', max(avg_signal) * 0.5, ...
                             'MinPeakDistance', round(0.6 * sampling_rate));
end

function [bvp_peaks, bvp_filtered] = bvp_findpeaks(bvp_signal, sampling_rate)
    % Find peaks in the BVP signal
    % Inputs:
    % - bvp_signal: Cleaned BVP signal
    % - sampling_rate: Sampling rate (Hz)
    % Outputs:
    % - bvp_peaks: Indices of detected BVP peaks
    % - bvp_filtered: Filtered BVP signal

    % Bandpass filter the BVP signal between 0.5 and 8 Hz
    bvp_filtered = bandpass(bvp_signal, [0.5, 8], sampling_rate);

    % Detect peaks in the BVP signal corresponding to heartbeats
    [~, bvp_peaks] = findpeaks(bvp_filtered, 'MinPeakHeight', max(bvp_filtered) * 0.5, ...
                               'MinPeakDistance', round(0.6 * sampling_rate));
end

function entropy = approximate_entropy(signal)
    % Compute approximate entropy of a signal
    % Inputs:
    % - signal: Time0