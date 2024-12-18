%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     Pre-Processing P300 Signal Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     Input:                 'Subject_A_train.mat' or 'Subject_B_train.mat'
%%%%%     Output:              Result: a matrix with dimensions (85,12,15,60,33)
%%%%%                                  labelR: a matrix with dimensions (85,12,15)        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

%% Initialization
tic;
str = 'Subject_A_Train.mat'; % Input file

load(str); % Load data file
fprintf('INFO: Data loaded and pre-processing started...\n\n');

% Convert data to double precision
Signal = double(Signal);
Flashing = double(Flashing);
StimulusCode = double(StimulusCode);
StimulusType = double(StimulusType);

%% Parameter Initialization
feature_dim = 60; % Feature dimension
numTrials = size(Signal, 1); % Number of trials
numChars = 12; % Number of characters
numRepeats = 15; % Number of repeats
numSamples = 240; % Total sample size
channel = 1:64; % Channel range
numChannels = length(channel); % Number of channels
matrix = ['ABCDEF'; 'GHIJKL'; 'MNOPQR'; 'STUVWX'; 'YZ1234'; '56789_']; % Character matrix
full_scale = 240; % Original sampling rate
down_sample_scale = 4; % Downsampling scale

numSamples = 0.6 * full_scale; % Update sample size
numSamplesUsed = numSamples / down_sample_scale; % Downsampled sample size
numFeatures = numSamplesUsed * numChannels; % Total features
numUsedChannels = length(channel);

%% Filter Design
order = 10; % Filter order
fstop1 = 0; % First stopband frequency
fpass1 = 0.5; % First passband frequency
fpass2 = 20; % Second passband frequency
fstop2 = 21; % Second stopband frequency
wstop1 = 1; % First stopband weight
wpass = 1; % Passband weight
wstop2 = 2; % Second stopband weight
dens = 20; % Density factor

% Create filter coefficients using the Parks-McClellan algorithm
b = firpm(order, [0 fstop1 fpass1 fpass2 fstop2 full_scale/2]/(full_scale/2), ...
    [0 0 1 1 0 0], [wstop1 wpass wstop2], {dens});
Hd = dfilt.dffir(b); % Create filter object

%% Filtering
Signal_filtered = zeros(size(Signal));
for i = 1:numTrials
    Signal_trial = squeeze(Signal(i, :, :));
    Signal_filtered(i, :, :) = reshape(filter(Hd, Signal_trial), 1, size(Signal_trial, 1), size(Signal_trial, 2));
end
fprintf('INFO: Filtering complete...\n\n');

%% Preprocessing Data
featureTrain = [];
labelTrain = [];

for epoch = 1:numTrials
    repeat = zeros(1, numChars);
    signalTrial = zeros(numChars, numRepeats, numSamples, numChannels);
    fprintf('INFO: Processing trial %d / %d...\n', epoch, numTrials);

    % Extract signal segments
    for n = 2:size(Signal, 2)
        if Flashing(epoch, n) == 1 && Flashing(epoch, n-1) == 0
            event = StimulusCode(epoch, n);
            repeat(event) = repeat(event) + 1;
            signalTrial(event, repeat(event), :, :) = Signal_filtered(epoch, n:n+numSamples-1, :); 
        end
    end

    % Process downsampled and normalized signals
    featureTrial = zeros(numChars, numRepeats, numFeatures);
    for char = 1:numChars
        for repeat_1 = 1:numRepeats
            signalFiltered = squeeze(signalTrial(char, repeat_1, :, :));
            signalDownsampled = downsample(signalFiltered, down_sample_scale);
            for c = 1:numUsedChannels
                signalNormalized(:, c) = zscore(signalDownsampled(:, c));
            end
            featureTrial(char, repeat_1, :) = signalNormalized(:);
        end
    end

    % Reshape and generate labels
    featureTrial = reshape(featureTrial, numChars*numRepeats, numSamplesUsed, numChannels);
    featureTrain = cat(1, featureTrain, featureTrial);
    
    targetIndex = strfind(matrix, TargetChar(epoch));
    targetRow = floor((targetIndex-1)/6) + 1;
    targetCol = targetIndex - (targetRow-1)*6;
    labelTrial = zeros(numChars, 1);
    labelTrial([targetCol, targetRow+6]) = 1;
    labelTrain = cat(1, labelTrain, repmat(labelTrial, numRepeats, 1));
end

% Convert to single precision
Dtrain = single(featureTrain);
Ltrain = single(labelTrain);

% Save processed data
fprintf('INFO: Preprocessing finished. Saving data...\n');
save TrainData Dtrain Ltrain
toc;
