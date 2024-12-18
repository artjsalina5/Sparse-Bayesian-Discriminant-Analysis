# Sparse Bayesian Discriminant Analysis (SBDA)

## Overview
This project implements and applies Sparse Bayesian Discriminant Analysis (SBDA) for feature selection and classification in EEG signal processing. SBDA is a powerful machine learning method that combines Bayesian inference with sparse modeling to achieve robust classification while reducing overfitting. The method is particularly suited for high-dimensional datasets with relatively small sample sizes, such as EEG signals.

## Contents

### Directory Structure
```
├── Subject_A_Train.mat      # Training EEG data
├── Subject_A_Test.mat       # Testing EEG data
├── TrainData.mat            # Processed training data
├── TestData.mat             # Processed testing data
├── windsorize.m             # Function to apply Windsorization
├── SBDA.m                   # Full implementation of Sparse Bayesian Discriminant Analysis
├── SBDA_easy.m              # Sparse Bayesian Discriminant Analysis implementation
├── fcs.m                    # Fisher Criterion Score for feature selection
├── extract.m                # Feature extraction function
├── reduction.m              # Dimensionality reduction function
├── train_preprocessing.m    # Training data preprocessing script
├── test_preprocessing.m     # Testing data preprocessing script
└── README.md                # Project documentation
```

### Scripts

#### `SBDA.m`
- Implements the full version of Sparse Bayesian Discriminant Analysis.
- Includes group-level feature handling, iterative updates of weights, and hyperparameters (`alpha`, `beta`).
- Provides detailed convergence criteria based on evidence and weight changes.
- Suitable for datasets with hierarchical or grouped feature structures.
  
#### `SBDA_easy.m`
- Implements a simplified version of Sparse Bayesian Discriminant Analysis.
- Iteratively updates weights (`w`), hyperparameters (`alpha` and `beta`), and evidence to identify the most relevant features for classification.
- Provides efficient handling of high-dimensional feature spaces by enforcing sparsity in the solution.

#### `fcs.m`
- Computes the Fisher Criterion Score for selecting features that maximize class separability.

#### `train_preprocessing.m`
- Loads the training dataset (`Subject_A_Train.mat`).
- Applies FIR filtering to raw EEG signals.
- Extracts signal windows around stimulus events.
- Normalizes and downsamples the signals.
- Generates labels and saves the processed training data to `TrainData.mat`.

#### `test_preprocessing.m`
- Loads the testing dataset (`Subject_A_Test.mat`).
- Processes the data similarly to the training script.
- Saves the processed testing data to `TestData.mat`.

#### `windsorize.m`
- Removes outliers by clamping values to the 10th and 90th percentiles.
- Prepares the data for further processing or feature extraction.

#### `extract.m`
- Extracts features from the processed EEG data.

#### `reduction.m`
- Reduces the dimensionality of features based on selected indices.

## Sparse Bayesian Discriminant Analysis (SBDA)

### Theory
SBDA is a probabilistic approach to discriminant analysis that incorporates Bayesian inference to estimate class boundaries. The algorithm leverages sparsity-inducing priors on the feature weights, encouraging most weights to shrink to zero while retaining only the most informative features.

#### Key Steps:
1. **Initialization:**
   - Initialize weights (`w`), precision parameters (`alpha`), and noise variance (`beta`).
2. **Iterative Updates:**
   - Update weights based on the current estimates of `alpha` and `beta`.
   - Update `alpha` to encourage sparsity in feature weights.
   - Update `beta` based on the residual error between predictions and ground truth.
3. **Convergence:**
   - Stop iterations when changes in evidence or weights fall below a predefined threshold.

### Advantages
- **Sparsity:** Automatically selects the most relevant features, reducing overfitting.
- **Bayesian Framework:** Provides a probabilistic interpretation of model parameters.
- **Efficiency:** Scales well to high-dimensional datasets.

### Application in EEG Data
- **High Dimensionality:** EEG signals often have many channels and time points, making SBDA an ideal choice for feature selection.
- **Noise Robustness:** The Bayesian framework inherently handles noise in the data.

### Implementation
The provided `SBDA_easy.m` script simplifies the SBDA algorithm while retaining its core functionality. It iteratively optimizes the evidence and hyperparameters to achieve an optimal feature subset.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/p300-signal-processing.git
   cd p300-signal-processing
   ```
2. Ensure MATLAB is installed with the required toolboxes:
   - Signal Processing Toolbox
   - Statistics and Machine Learning Toolbox

## Usage

### Preprocessing Training Data
Run the following command in MATLAB:
```matlab
train_preprocessing;
```

### Preprocessing Testing Data
Run the following command in MATLAB:
```matlab
test_preprocessing;
```

### Feature Selection with SBDA
Use the `SBDA_easy.m` script to perform feature selection and classification:
```matlab
model = SBDA_easy(y, X);
```

### Windsorization
To remove outliers from the dataset:
```matlab
RR = windsorize(R);
```

## Project Details
- **Language:** MATLAB
- **Data:** EEG signals from the `Subject_A_Train.mat` and `Subject_A_Test.mat` files.
- **Goal:** To extract and prepare features for classification of P300 signals using SBDA.

## References
- [Sparse Bayesian Learning](https://en.wikipedia.org/wiki/Sparse_Bayesian_learning)
- [P300 Component](https://en.wikipedia.org/wiki/P300_(neuroscience))

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
Special thanks to the team and contributors for providing data and tools for EEG analysis.
