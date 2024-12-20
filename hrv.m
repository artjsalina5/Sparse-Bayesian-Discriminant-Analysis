% Initialize Python environment
if count(py.sys.path,'') == 0
    insert(py.sys.path, int32(0), '');
end

% Import NeuroKit2
nk = py.importlib.import_module('neurokit2');

% Define parameters
sampling_rate = 256; % Hz
window_size = 60; % seconds

% Clean ECG signal and find R-peaks
ecg_cleaned = nk.ecg_clean(data{'ECG'}, pyargs('sampling_rate', sampling_rate));
[peaks, ~] = nk.ecg_peaks(ecg_cleaned, pyargs('sampling_rate', sampling_rate));

% Extract R-peak indices
rpeaks = peaks{'ECG_R_Peaks'};

% Determine the number of windows
num_samples_per_window = sampling_rate * window_size;
num_windows = int32(length(data) / num_samples_per_window);

% Prepare results storage
hrv_results = {};

% Loop through each 60-second window
for i = 1:num_windows
    % Define start and end indices for the window
    start_idx = (i - 1) * num_samples_per_window + 1;
    end_idx = i * num_samples_per_window;

    % Get R-peaks in the current window
    window_rpeaks = py.numpy.array(rpeaks(rpeaks >= start_idx & rpeaks < end_idx));

    % Skip if not enough data
    if length(window_rpeaks) < 2
        continue;
    end

    % Calculate HRV metrics
    rri = py.numpy.diff(window_rpeaks) / sampling_rate;
    rri_time = window_rpeaks(2:end) / sampling_rate;

    hrv_metrics = nk.hrv(py.dict(pyargs(...
        'RRI', rri, ...
        'RRI_Time', rri_time ...
    )), pyargs('sampling_rate', sampling_rate));

    % Store HRV results
    hrv_results{end+1} = hrv_metrics; %#ok<SAGROW>
end

% Convert results to MATLAB table
results_table = [];
for k = 1:length(hrv_results)
    py_table = hrv_results{k};
    results_table = [results_table; struct(py_table)] %#ok<AGROW>
end

% Display the HRV results
disp(results_table);