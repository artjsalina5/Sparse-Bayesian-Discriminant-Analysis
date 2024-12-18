%{
Windsorize: Preprocesses the data by applying Windsorization to remove outliers.
Keeps data within the 10th and 90th percentiles by replacing extreme values.

Input:
R - 3D matrix (no x window x ch), where:
    no: Number of trials
    window: Number of samples in each trial
    ch: Number of channels

Output:
RR - 3D matrix after Windsorization
%}
function RR = windsorize(R)
    [no, window, ch] = size(R); % Dimensions of input matrix
    D = zeros(no * window, ch); % Flattened data matrix (for processing)
    RR = zeros(no, window, ch); % Preprocessed output matrix
    
    % Flatten the data from 3D to 2D for processing
    for i = 1:ch
        for j = 1:no
            D((j-1)*window + 1:j*window, i) = R(j, :, i);
        end
    end

    % Calculate thresholds for Windsorization
    lowerLimitIdx = round(0.1 * no * window); % 10th percentile index
    upperLimitIdx = round(0.9 * no * window); % 90th percentile index
    lowerLimitVals = zeros(ch, 1); % Storage for 10th percentile values
    upperLimitVals = zeros(ch, 1); % Storage for 90th percentile values
    
    % Sort data along each channel
    [~, idx] = sort(D, 'descend'); 

    % Find the 10th and 90th percentile values for each channel
    for ii = 1:ch
        lowerLimitVals(ii) = D(idx(lowerLimitIdx, ii), ii);
        upperLimitVals(ii) = D(idx(upperLimitIdx, ii), ii);
    end
    
    % Apply Windsorization: Replace values outside the range
    for iii = 1:ch
        for jjj = 1:no * window
            if D(jjj, iii) >= upperLimitVals(iii)
                D(jjj, iii) = upperLimitVals(iii);
            elseif D(jjj, iii) <= lowerLimitVals(iii)
                D(jjj, iii) = lowerLimitVals(iii);
            end
        end
    end

    % Reshape the processed data back to the original 3D format
    for iiii = 1:ch
        for jjjj = 1:no
            RR(jjjj, :, iiii) = D((jjjj-1)*window + 1:jjjj*window, iiii);
        end
    end
end
