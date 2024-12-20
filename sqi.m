%% Robust BVP Signal Processing with SQI, ECG Reference, and Interpolation
% This script preprocesses a noisy BVP signal using SQI, aligns it with a clean ECG signal,
% and replaces invalid segments using cubic spline interpolation.

clc; clear;

%% Parameters
Fs = 256; % Sampling frequency (Hz)
bvp_filter_cutoff = 15; % Low-pass filter cutoff frequency (Hz)
bpm_range = [40, 180]; % Valid HR range in bpm
window_length = 10 * Fs; % 10-second window in samples

%% Load Signals
% Replace these placeholders with actual ECG and BVP signals
load('ECG_signal.mat'); % Clean ECG signal
load('BVP_signal.mat'); % Noisy BVP signal

% ECG R-peak detection
[~, r_peaks] = preprocessECG(ECG_signal, Fs);

%% Preprocess and Validate BVP
% Low-pass filter BVP
filtered_bvp = lowpass(BVP_signal, bvp_filter_cutoff, Fs);

% Validate BVP based on proximity to ECG R-peaks
[clean_BVP, sqi_flags] = validateAndInterpolateBVP(filtered_bvp, r_peaks, bpm_range, Fs);

%% Plot Results
figure;
subplot(3, 1, 1);
plot(ECG_signal);
title('ECG Signal (Clean)');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(BVP_signal);
title('BVP Signal (Raw)');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3, 1, 3);
plot(clean_BVP, 'g');
title('BVP Signal (Validated and Interpolated)');
xlabel('Time (s)');
ylabel('Amplitude');

%% Save Validated and Interpolated BVP Signal
save('Validated_BVP.mat', 'clean_BVP');

%% Supporting Functions

function [filtered_ecg, r_peaks] = preprocessECG(ecg_signal, fs)
    % Preprocess ECG signal and detect R-peaks.
    bp_filter = designfilt('bandpassiir', 'FilterOrder', 4, ...
        'HalfPowerFrequency1', 0.5, 'HalfPowerFrequency2', 40, ...
        'SampleRate', fs);
    filtered_ecg = filtfilt(bp_filter, ecg_signal);
    [~, r_peaks] = findpeaks(filtered_ecg, 'MinPeakHeight', mean(filtered_ecg) + std(filtered_ecg));
end

function [clean_BVP, sqi_flags] = validateAndInterpolateBVP(bvp_signal, r_peaks, bpm_range, fs)
    % Validate and clean BVP signal using SQI, aligning with ECG R-peaks.
    % Interpolates invalid segments using cubic spline.
    %
    % Inputs:
    % - bvp_signal: Filtered BVP signal
    % - r_peaks: Indices of R-peaks in the ECG signal
    % - bpm_range: Valid HR range in bpm
    % - fs: Sampling frequency (Hz)
    %
    % Outputs:
    % - clean_BVP: Validated and interpolated BVP signal
    % - sqi_flags: Binary flags indicating valid segments

    % Initialize variables
    clean_BVP = bvp_signal;
    sqi_flags = false(length(bvp_signal), 1);

    % Adaptive threshold for BVP peak detection
    [bvp_peaks, ~] = findpeaks(bvp_signal, ...
        'MinPeakHeight', mean(bvp_signal) + 0.5 * std(bvp_signal));

    % Validate peaks based on ECG R-peaks
    valid_peaks = [];
    for i = 1:length(r_peaks)-1
        % Define time window between consecutive R-peaks
        window_start = r_peaks(i);
        window_end = r_peaks(i+1);
        window_peaks = bvp_peaks(bvp_peaks > window_start & bvp_peaks < window_end);

        % Validate HR range
        if isempty(window_peaks)
            continue;
        end
        ibi = diff(window_peaks) / fs; % Inter-Beat Intervals (seconds)
        ibi_bpm = 60 ./ ibi; % Convert to bpm

        if all(ibi_bpm >= bpm_range(1) & ibi_bpm <= bpm_range(2))
            % Mark valid peaks and corresponding BVP segment
            valid_peaks = [valid_peaks; window_peaks];
            sqi_flags(window_peaks) = true;
        end
    end

    % Mark invalid values and interpolate
    invalid_idx = setdiff(1:length(bvp_signal), valid_peaks);
    clean_BVP(invalid_idx) = NaN; % Mark invalid values as NaN
    valid_idx = find(~isnan(clean_BVP));
    clean_BVP = interp1(valid_idx, clean_BVP(valid_idx), 1:length(clean_BVP), 'pchip'); % Cubic spline interpolation
end