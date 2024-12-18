clear;
tic;

% Load test data
str = 'Subject_A_Test.mat';
load(str);
fprintf('INFO: Test data loaded. Starting pre-processing...\n\n');

% Convert to double precision
Signal = double(Signal);
Flashing = double(Flashing);
StimulusCode = double(StimulusCode);

%% Parameters
full_scale = 240; % Sampling frequency
numTrials = size(Signal, 1);
numChars = 12;
numRepeats = 15;
channel = 1:64; % Channels to use
numChannels = length(channel);
matrix = ['ABCDEF'; 'GHIJKL'; 'MNOPQR'; 'STUVWX'; 'YZ1234'; '56789_'];
featureTrain = [];
labelTest = [];

% Target character sequence
target = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU';
targetTrue = zeros(numTrials, 1);
for i = 1:numTrials
    targetTrue(i) = target(i);
end

%% Signal parameters
numSamples = 0.6 * full_scale; % 0.6 seconds of data
down_sample_scale = 4; % Downsampling factor
numUsedSamples = numSamples / down_sample_scale;
numFeatures = numUsedSamples * numChannels;

%% Filter design
order = 10;
fstop1 = 0;    % First Stopband Frequency
fpass1 = 0.5;  % First Passband Frequency
fpass2 = 20;   % Second Passband Frequency
fstop2 = 21;   % Second Stopband Frequency
wstop1 = 1;    % First Stopband Weight
wpass = 1;     % Passband Weight
wstop2 = 2;    % Second Stopband Weight
dens = 20;     % Density Factor

% Design FIR filter
b = firpm(order, [0 fstop1 fpass1 fpass2 fstop2 full_scale/2] / (full_scale/2), [0 0 1 1 0 0], [wstop1 wpass wstop2], {dens});
Hd = dfilt.dffir(b);

% Filter the signal
Signal_filtered = zeros(size(Signal));
for i = 1:numTrials
    Signal_trial = squeeze(Signal(i, :, :));
    Signal_filtered(i, :, :) = reshape(filter(Hd, Signal_trial), 1, size(Signal_trial, 1), size(Signal_trial, 2));
end
fprintf('INFO: Signal filtering complete.\n\n');

%% Process each trial
for epoch = 1:numTrials
    repeat = zeros(1, numChars);
    signalTrial = zeros(numChars, numRepeats, numSamples, numChannels);
    featureTrial = zeros(numChars, numRepeats, numUsedSamples, numChannels);

    fprintf('INFO: Processing epoch %d / %d...\n', epoch, numTrials);

    % Extract trial segments where flashing starts
    for n = 2:size(Signal, 2)
        if Flashing(epoch, n - 1) == 0 && Flashing(epoch, n) == 1
            event = StimulusCode(epoch, n);
            repeat(event) = repeat(event) + 1;
            signalTrial(event, repeat(event), :, :) = Signal_filtered(epoch, n:n + numSamples - 1, :);
        end
    end

    % Downsample and normalize the signal
    for char = 1:numChars
        for repeat_1 = 1:numRepeats
            signalFiltered = squeeze(signalTrial(char, repeat_1, :, :));
            signalDownsampled = downsample(signalFiltered, down_sample_scale);
            for c = 1:numChannels
                featureTrial(char, repeat_1, :, c) = zscore(signalDownsampled(:, c));
            end
        end
    end

    % Reshape features and concatenate into featureTrain
    featureTrial = reshape(featureTrial, numChars * numRepeats, numUsedSamples, numChannels);
    featureTrain = cat(1, featureTrain, featureTrial);

    % Generate label matrix
    targetIndex = strfind(matrix, target(epoch));
    targetRow = floor((targetIndex - 1) / 6) + 1;
    targetCol = targetIndex - (targetRow - 1) * 6;
    labelTrial = zeros(numChars, 1);
    labelTrial([targetCol, targetRow + 6]) = 1;
    labelTest = cat(1, labelTest, repmat(labelTrial, numRepeats, 1));
end

%% Save processed data
Dtest = single(featureTrain);
Ltest = single(labelTest);

save TestData Dtest Ltest targetTrue;

toc;
