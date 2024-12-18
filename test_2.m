function test_2(feature_dim)
    tic;
    load('TrainData.mat'); % Load training data
    load('TestData.mat');  % Load testing data
    matrix = ['ABCDEF'; 'GHIJKL'; 'MNOPQR'; 'STUVWX'; 'YZ1234'; '56789_']; % Target character matrix

    % Prepare training data
    featureTrain = double(Dtrain);
    labelTrain = double(Ltrain);
    clear Dtrain; % Free memory
    clear Ltrain;

    % Initialize parameters
    numChars = 12;          % Number of characters
    numRepeats = 15;        % Number of repetitions
    numSamples = size(featureTrain, 2); % Number of samples
    numChannels = size(featureTrain, 3); % Number of channels
    numTrain = size(featureTrain, 1) / (numChars * numRepeats); % Number of training trials
    groupChannel = reshape(repmat(1:numChannels, numSamples, 1), 1, numChannels * numSamples);

    fprintf('INFO: Training...\n\n');
    X = featureTrain;
    X = reshape(X, size(X, 1), size(X, 2) * size(X, 3)); % Reshape data for training
    X = svmscale(X, [0 1], 'range', 's'); % Scale data
    y = labelTrain;
    y(y == 0) = -1; % Convert labels to binary format (-1, 1)
    clear featureTrain; % Free memory
    clear labelTrain;

    % Feature selection using FCS (Fisher criterion selection)
    X_2d = X(:, :);
    idxp300 = find(y == 1); % Positive class indices
    idxnp300 = find(y == -1); % Negative class indices
    idx_fcs = fcs(X(idxp300, :), X(idxnp300, :)); % Feature selection
    X = reduction(X_2d, idx_fcs, feature_dim); % Reduce features to specified dimension

    % Train the model
    train_model = SBDA_easy(y, X);  
    fprintf('INFO: Training completed.\n\n');

    % Prepare test data
    featureTest = double(Dtest);
    labelTest = double(Ltest);
    numTest = size(featureTest, 1) / (numChars * numRepeats);

    fprintf('INFO: Classifying...\n\n');
    X = featureTest;
    X = reshape(X, size(X, 1), size(X, 2) * size(X, 3)); % Reshape data for testing
    X = svmscale(X, [0 1], 'range', 'r'); % Scale data
    y = labelTest;
    y(y == 0) = -1; % Convert labels to binary format (-1, 1)
    clear featureTest; % Free memory
    clear labelTest;

    % Reduce features for test data
    X_2d = X(:, :);
    X = reduction(X_2d, idx_fcs, feature_dim);

    % Predictions
    yprob = X * train_model.b + train_model.b0; % Compute probabilities
    ypred = sign(yprob); % Classify based on sign

    % Confusion matrix
    idxp = find(y == 1); % Positive indices
    idxn = find(y == -1); % Negative indices

    TP = length(find(ypred(idxp) == 1)); % True positives
    FP = length(find(ypred(idxn) == 1)); % False positives
    TN = length(find(ypred(idxn) == -1)); % True negatives
    FN = length(find(ypred(idxp) == -1)); % False negatives
    confusion = [TP, TN, FP, FN];

    fprintf('INFO: Confusion Matrix:\n');
    fprintf('TP: %d, TN: %d, FP: %d, FN: %d\n\n', TP, TN, FP, FN);

    % Compute target predictions and validation accuracies
    targetPredicted = zeros(numRepeats, numTest);
    for trial = 1:numTest
        yprob1 = yprob(:, 1);
        ytrial = yprob1((trial-1)*numChars*numRepeats + (1:numChars*numRepeats));
        ytrial = reshape(ytrial, numChars, numRepeats);
        for repeat = 1:numRepeats
            yavg = mean(ytrial(:, 1:repeat), 2);
            [~, pRow] = max(yavg(7:12)); % Row prediction
            [~, pCol] = max(yavg(1:6)); % Column prediction
            targetPredicted(repeat, trial) = matrix((pRow-1)*6 + pCol);
        end
    end

    % Compute accuracies
    for j = 1:numRepeats
        accuracyTest(j) = length(find(squeeze(targetPredicted(j, :)) == targetTrue')) / numTest;
        fprintf('INFO: Validation accuracy after %d repetitions: %.2f%%\n', j, accuracyTest(j) * 100);
    end

    % Display results
    disp('Results displayed.');
    toc;
end
