# Peak Detection Module

This module provides tools for generating synthetic ECG signals with various signal-to-noise ratios (SNR) and evaluating different peak detection methods.

## Files

### 1. `generate_synthetic_ecgs.py`

This script generates synthetic ECG signals with different SNR levels and evaluates the performance of multiple peak detection methods.

**Features:**
- Generates synthetic ECG signals using McSharry's model
- Adds controlled noise with specific SNR levels `(20, 15, 10, 7, 5, 3, 2, 1, 0.8, 0.5, 0.1)`
- Tests 9 different peak detection methods:
  - `nabian2018`
  - `pantompkins1985`
  - `hamilton2002`
  - `elgendi2010`
  - `engzeemod2012`
  - `kalidas2017`
  - `martinez2004`
  - `rodrigues2021`
  - `manikandan2012`
- Computes performance metrics for each method and SNR level
- Performs statistical analysis (ANOVA) to compare methods
- Saves results to a pickle file for further analysis

**How to run:**
```bash
python generate_synthetic_ecgs.py
```
An example log file is provided in `example_log_generate_syn_ecgs.txt`. 

### 2. `create_visualization.py`

This script creates visualizations of the peak detection results, generating heatmaps for various metrics.

**Features:**
- Creates heatmap visualizations for different metrics:
  - F1 Score
  - Sensitivity
  - Precision
  - Mean Residuals
  - Variance of Residuals
  - Normality Test p-values
  - Wilcoxon Test p-values
- Saves visualizations to the `visualization_results` directory

**How to run:**
```bash
python create_visualization.py
```
All the graphics produced are saved in the `data/visualization_results/` directory so that you can skip running the script yourself.

## Dependencies

This module requires the following dependencies, which are included in the main environment.yaml file:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- neurokit2
- tqdm

## Directory Structure

```
peak_detection/
├── data/                  # Directory for data files
│   └── ecg_noise_template.pkl  # Template for noise generation
│   └── synthetic_results/      # Directory for generated ECG signals
│   └── visualization_results/  # Directory for visualization outputs
├── generate_synthetic_ecgs.py  # Script for generating synthetic ECGs
├── create_visualization.py     # Script for creating visualizations
```

## Workflow
0. Activate the conda environment using `conda activate hrv_suite`.
1. First, run `generate_synthetic_ecgs.py` to generate synthetic ECG signals and evaluate peak detection methods.
2. Then, run `create_visualization.py` to create visualizations of the results
