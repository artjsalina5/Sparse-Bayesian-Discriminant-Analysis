% Change directory if needed
% cd('Y:\Graduation Project\test_2');
clc;
clear all;

% Load data
load('Subject_A_Train.mat');

%% Parameters
windows = 200;       % Window size after stimulus (240Hz sampling rate, resampled to 32Hz)
feature_dim = 500;   % Dimension for feature reduction
tic;

% Display initial processing message
fprintf('INFO: Data loaded. Starting pre-processing...\n\n');

% Convert data to double precision
Signal = double(Signal);
Flashing = double(Flashing);
StimulusCode = double(StimulusCode);
StimulusType = double(StimulusType);

% Initialize response arrays
responses_0 = zeros(size(Signal, 1) * 12 * 15, windows, 64);
responses_1 = zeros(size(Signal, 1) * 12 * 15, windows / 4, 64);
responses_3 = zeros(size(Signal, 1) * 12 * 15, windows, 64);
label = zeros(size(Signal, 1) * 12 * 15);
chorespT = zeros(size(Signal, 1) * 12 * 15 / 6, windows / 4, 64);
chorespNT = zeros(size(Signal, 1) * 12 * 15 * 5 / 6, windows / 4, 64);

% Counters
count_1 = 0;
count_2 = 0;
count_3 = 0;

%% Filter Parameters
fs = 240; % Sampling rate
order = 10;
fstop1 = 0; fpass1 = 0.5; % Frequency ranges
fpass2 = 20; fstop2 = 21;
wstop1 = 1; wpass = 1; wstop2 = 2; % Weight factors
dens = 20; % Density factor

% Design the FIR filter using the Parks-McClellan algorithm
b = firpm(order, [0 fstop1 fpass1 fpass2 fstop2 fs / 2] / (fs / 2), ...
    [0 0 1 1 0 0], [wstop1 wpass wstop2], {dens});
Hd = dfilt.dffir(b);

% Apply filtering
numTrials = size(Signal, 1);
Signal_filtered = zeros(size(Signal));

for i = 1:numTrials
    Signal_trial = squeeze(Signal(i, :, :));
    Signal_filtered(i, :, :) = reshape(filter(Hd, Signal_trial), ...
        1, size(Signal_trial, 1), size(Signal_trial, 2));
end
fprintf('INFO: Filtering complete.\n\n');

%% Extract Responses
for epoch = 1:size(Signal, 1) % Process each trial
    fprintf('INFO: Processing trial %d of %d...\n', epoch, size(Signal, 1));
    
    % Baseline correction for each channel
    for channel = 1:64
        % Reference channels T7 (41) and T8 (42)
        if channel ~= 41 && channel ~= 42
            Signal(epoch, :, channel) = Signal_filtered(epoch, :, channel) ...
                - 0.5 * Signal_filtered(epoch, :, 41) ...
                - 0.5 * Signal_filtered(epoch, :, 42);
        end
    end

    % Process events
    for n = 2:size(Signal, 2)
        % Detect onset of a flashing stimulus
        if Flashing(epoch, n) == 0 && Flashing(epoch, n - 1) == 1
            count_1 = count_1 + 1;

            % Extract response window
            responses_0(count_1, :, :) = Signal(epoch, n - 24:n - 25 + windows, :);
            
            % Downsample responses
            for j = 1:64
                responses_1(count_1, :, j) = responses_0(count_1, 1:4:end, j);
            end
            
            % Assign labels
            code(count_1) = StimulusCode(epoch, n - 1);
            label(count_1) = StimulusType(epoch, n - 1);

            % Separate P300 and non-P300 responses
            if label(count_1) == 1
                count_2 = count_2 + 1;
                chorespT(count_2, :, :) = responses_1(count_1, :, :);
            else
                count_3 = count_3 + 1;
                chorespNT(count_3, :, :) = responses_1(count_1, :, :);
            end
        end
    end
end

fprintf('INFO: Pre-processing complete.\n\n');

%% Feature Extraction and Reduction
DP = extract(chorespT); % Extract P300 responses
DN = extract(chorespNT); % Extract non-P300 responses

% Feature selection using FCS
[~, idx] = fcs(DP, DN);
DPnew = reduction(DP, idx, feature_dim); % Reduce dimensions for P300
DNnew = reduction(DN, idx, feature_dim); % Reduce dimensions for non-P300

toc;
