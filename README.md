# HRV Processing Suite

This is the master repository providing all necessary code for the paper titled "Detection of Diabetic Peripheral Neuropathy through Heart Rate Variability"

## Project Structure
The repository is organized into three main folders
```
.
├── diabeat/          # Deep Learning Model for HRV Analysis
└── peak_detection/   # Peak Detection and Synthetic Data Generation
```

## Setup
Clone and setup the repository:
```bash
git clone https://github.com/ad-aspera/hrv_suite.git && git submodule update --init --recursive
```

## Dependencies

Createa a conda environment using the `environment.yaml` file by running the following command:
```bash
conda env create -f environment.yaml
```
This should install all the necessary dependencies.

## Usage

Please refer to the README files in each subfolder for usage instructions.

- [diabeat/README.md](diabeat/README.md) - Instructions for training/running the deep learning model.
- [peak_detection/README.md](peak_detection/README.md) - Instructions for peak detection and synthetic data generation methods.