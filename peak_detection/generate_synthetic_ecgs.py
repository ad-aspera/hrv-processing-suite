#!/usr/bin/env python
"""
Module for generating and saving synthetic ECG data with various SNR levels.

This module provides functions to:
1. Generate synthetic ECG signals using McSharry's model
2. Add controlled noise with specific SNR levels
3. Save the generated signals and evaluate peak detection performance
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import neurokit2 as nk
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
from scipy.stats import shapiro, wilcoxon, f_oneway
from tqdm import tqdm

# Configure matplotlib
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"

# Filter out warnings from statistical tests
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")


def load_noise_template(template_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load the noise template from a pickle file.

    Args:
        template_path (Union[str, Path]): Path to the pickle file containing the noise template.
                                         The template should be a dictionary with at least a
                                         'freq_template' key containing a numpy array.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing the noise template with frequency domain data.
    """
    with open(template_path, "rb") as f:
        return pickle.load(f)


def generate_noise(
    template: Dict[str, np.ndarray],
    target_length: int,
    target_power: Optional[float] = None,
) -> np.ndarray:
    """
    Generate noise based on the provided frequency template.

    Args:
        template (Dict[str, np.ndarray]): Dictionary containing the noise template with at least a
                                         'freq_template' key containing a numpy array.
        target_length (int): Desired length of the generated noise signal.
        target_power (Optional[float]): Target power level for the generated noise. If provided,
                                      the noise will be scaled to match this power level.

    Returns:
        np.ndarray: Generated noise signal in the time domain.
    """
    freq_template = template["freq_template"]

    if freq_template.ndim > 1:
        freq_template = freq_template[:, 0]

    # Adjust template length if needed
    if len(freq_template) != target_length:
        x = np.linspace(0, 1, len(freq_template))
        x_new = np.linspace(0, 1, target_length)
        freq_template = np.interp(x_new, x, freq_template)

    # Add random phase and perform inverse FFT to create time-domain noise
    random_phase = np.exp(1j * 2 * np.pi * np.random.random(target_length))
    complex_spectrum = freq_template * random_phase
    noise = np.real(ifft(complex_spectrum))

    # Adjust power if target_power is provided
    if target_power is not None:
        current_power = np.mean(np.square(noise))
        scale_factor = np.sqrt(target_power / current_power)
        noise *= scale_factor

    return noise


def adjust_snr(signal: np.ndarray, noise: np.ndarray, target_snr: float) -> np.ndarray:
    """
    Scale noise to reach a specified signal-to-noise ratio (SNR).

    Args:
        signal (np.ndarray): The original clean signal.
        noise (np.ndarray): The noise signal to be scaled.
        target_snr (float): The desired signal-to-noise ratio. Higher values indicate less noise.

    Returns:
        np.ndarray: Scaled noise signal that, when added to the original signal,
                   will result in the specified SNR.
    """
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))
    target_noise_power = signal_power / target_snr
    scale_factor = np.sqrt(target_noise_power / noise_power)
    return noise * scale_factor


def generate_ecg_mcsharry(
    num_peaks: int = 30,
    fs: int = 250,
    heart_rate: int = 60,
    interval_variance: float = 0.2,
    target_amp: float = 5.0,
    amplitude_variance: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic ECG signal using McSharry's model.

    Args:
        num_peaks (int, optional): Number of R peaks to generate in the ECG signal. Defaults to 30.
        fs (int, optional): Sampling frequency in Hz. Defaults to 250.
        heart_rate (int, optional): Target heart rate in beats per minute. Defaults to 60.
        interval_variance (float, optional): Variance of the RR interval to introduce natural variability.
                                           Defaults to 0.2.
        target_amp (float, optional): Target amplitude for the R peaks. Defaults to 5.0.
        amplitude_variance (float, optional): Variance of the R peak amplitude to introduce natural variability.
                                            Defaults to 0.1.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - Time vector (in seconds)
            - ECG signal
            - Indices of true R peaks
    """
    dt = 1.0 / fs
    baseline_rr = 60.0 / heart_rate

    global_t = []
    global_ecg = []
    r_peak_indices = []
    current_time = 0.0
    sample_counter = 0

    # Define parameters for the PQRST waves.
    theta = np.zeros(5)
    a = np.zeros(5)
    b = np.zeros(5)
    theta[0] = -1 / 3 * np.pi  # P-wave
    theta[1] = -1 / 12 * np.pi  # Q-wave
    theta[2] = 0  # R-wave
    theta[3] = 1 / 12 * np.pi  # S-wave
    theta[4] = 1 / 2 * np.pi  # T-wave
    a[0] = 1.2
    a[1] = -5.0
    a[2] = 30.0
    a[3] = -7.5
    a[4] = 0.75
    b[0] = 0.25
    b[1] = 0.1
    b[2] = 0.1
    b[3] = 0.1
    b[4] = 0.4

    # Initial state on the unit circle.
    initial_angle = -2 / 3 * np.pi
    x_state = np.cos(initial_angle)
    y_state = np.sin(initial_angle)
    z_state = 0.0

    # Generate each heartbeat.
    for beat in range(num_peaks):
        rr = baseline_rr * (1 + np.random.normal(0, interval_variance))
        rr = max(rr, dt)
        N_beat = int(np.round(rr * fs))
        omega = 2 * np.pi / rr

        t_beat = np.linspace(0, rr, N_beat, endpoint=False)
        beat_z = np.zeros(N_beat)

        for i in range(N_beat):
            phi = np.arctan2(y_state, x_state)
            alpha = 1 - np.sqrt(x_state**2 + y_state**2)
            dx = alpha * x_state - omega * y_state
            dy = alpha * y_state + omega * x_state

            dz = 0
            for j in range(5):
                delta_theta = ((phi - theta[j]) + np.pi) % (2 * np.pi) - np.pi
                dz -= a[j] * delta_theta * np.exp(-(delta_theta**2) / (2 * b[j] ** 2))
            x_state = x_state + dt * dx
            y_state = y_state + dt * dy
            z_state = z_state + dt * dz

            beat_z[i] = z_state

        # Record the R-peak index.
        local_r_index = np.argmax(beat_z)
        r_peak_indices.append(sample_counter + local_r_index)

        # Scale beat amplitude to vary the R peak heights.
        local_r_amp = beat_z[local_r_index]
        desired_r_amp = np.random.normal(target_amp, amplitude_variance)
        scale_factor = desired_r_amp / local_r_amp if local_r_amp != 0 else 1.0
        beat_z = beat_z * scale_factor

        global_t.append(current_time + t_beat)
        global_ecg.append(beat_z)

        current_time += rr
        sample_counter += N_beat

        # Normalize (x_state, y_state) back to the unit circle.
        phi = np.arctan2(y_state, x_state)
        x_state = np.cos(phi)
        y_state = np.sin(phi)

    global_t = np.concatenate(global_t)
    global_ecg = np.concatenate(global_ecg)

    # Zero data before the first R peak.
    if r_peak_indices:
        start_point = max(0, r_peak_indices[0] - int(0.3 * fs))
        global_ecg[:start_point] = 0

    return global_t, global_ecg, np.array(r_peak_indices)


def add_random_deviation(
    signal: np.ndarray,
    fs: int = 250,
    dev_num: int = 1,
    component_num: int = 10,
    amp_var: float = 2.5,
    max_time: int = 700,
) -> np.ndarray:
    """
    Add random deviations to the ECG signal to simulate real-world variations.

    Args:
        signal (np.ndarray): The original ECG signal to modify.
        fs (int, optional): Sampling frequency in Hz. Defaults to 250.
        dev_num (int, optional): Number of random deviations to add. Defaults to 1.
        component_num (int, optional): Number of sinusoidal components in each deviation. Defaults to 10.
        amp_var (float, optional): Amplitude variance for the random deviations. Defaults to 2.5.
        max_time (int, optional): Maximum duration of each deviation in milliseconds. Defaults to 700.

    Returns:
        np.ndarray: ECG signal with random deviations added.
    """
    signal = signal.copy()
    for _ in range(dev_num):
        component = np.zeros(int(max_time * fs / 1000))
        time_scaler = max_time / 1000
        dev = component.copy()
        for _ in range(component_num):
            p = np.random.randint(1, 5)
            a = np.random.normal(0, amp_var)
            component = a * np.sin(
                p * np.pi * np.arange(len(component)) / fs / time_scaler
            )
            dev += component
        dev = (1 / component_num / 10) * dev
        dev_idx = np.random.randint(0, len(signal) - len(dev))
        signal[dev_idx : dev_idx + len(dev)] += dev
    return signal


def create_expanded_snr_dataset(
    output_dir: Union[str, Path],
    template_path: Union[str, Path],
    overwrite: bool = False,
    num_peaks: int = 1000,
    dev_prop: float = 0.05,
) -> np.ndarray:
    """
    Generate ECG signals at various SNR levels and save both the clean and noisy versions as CSV files.

    Args:
        output_dir (Union[str, Path]): Directory where the generated ECG signals will be saved.
        template_path (Union[str, Path]): Path to the noise template file.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        num_peaks (int, optional): Number of R peaks to generate in the clean ECG signal. Defaults to 1000.
        dev_prop (float, optional): Proportion of beats to add random deviations to. Defaults to 0.05.

    Returns:
        np.ndarray: Indices of the true R peaks in the generated ECG signal.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    snr_levels = [20, 15, 10, 7, 5, 3, 2, 1, 0.8, 0.5, 0.1]
    existing_files = [f for f in output_dir.glob("synthetic_ecg_snr_*.csv")]
    existing_snrs = []
    for f in existing_files:
        try:
            snr_str = f.stem.split("_")[-1]
            existing_snrs.append(float(snr_str))
        except ValueError:
            print(f"Warning: Could not parse SNR value from filename: {f}")

    if not overwrite:
        snr_levels = [snr for snr in snr_levels if snr not in existing_snrs]

    if not snr_levels:
        print(
            "All SNR levels already exist in the dataset. Use overwrite=True to regenerate."
        )
        return np.array([])

    print(f"Will generate ECG signals with SNR levels: {snr_levels}")

    try:
        noise_template = load_noise_template(template_path)
    except Exception as e:
        print(f"Error loading template: {e}")
        print("Using a basic noise profile instead")
        freq_template = np.ones(1000)
        noise_template = {"freq_template": freq_template, "avg_power": 1.0}

    clean_path = output_dir / "synthetic_ecg_clean.csv"
    if clean_path.exists():
        clean_df = pd.read_csv(clean_path)
        t = clean_df["time"].values
        clean_ecg = clean_df["ecg"].values

        fs = 1.0 / (t[1] - t[0])
        heart_rate = 80
        rr_samples = int((60.0 / heart_rate) * fs)
        height = 0.6 * np.max(clean_ecg)
        true_r_peaks, _ = find_peaks(
            clean_ecg, height=height, distance=rr_samples * 0.8
        )
    else:
        dev_num = int(dev_prop * num_peaks)
        fs = 500
        heart_rate = 80
        t, clean_ecg, true_r_peaks = generate_ecg_mcsharry(
            num_peaks=num_peaks, fs=fs, heart_rate=heart_rate
        )
        clean_ecg = add_random_deviation(
            clean_ecg,
            fs=fs,
            dev_num=dev_num,
            component_num=10,
            amp_var=2.5,
            max_time=700,
        )
        pd.DataFrame({"time": t, "ecg": clean_ecg}).to_csv(clean_path, index=False)

    for snr in tqdm(snr_levels, desc="Generating ECG with different SNR levels"):
        filename = f"synthetic_ecg_snr_{snr}.csv"
        output_path = output_dir / filename
        print(f"SNR={snr}, saving to {filename}")

        if not overwrite and output_path.exists():
            print(f"Skipping SNR={snr}, file already exists")
            continue

        noise = generate_noise(noise_template, len(clean_ecg))
        adjusted_noise = adjust_snr(clean_ecg, noise, snr)
        noisy_ecg = clean_ecg + adjusted_noise
        pd.DataFrame({"time": t, "ecg": noisy_ecg}).to_csv(output_path, index=False)

    np.save(output_dir / "true_r_peaks.npy", true_r_peaks)
    print(f"Generated ECG signals with SNR levels: {snr_levels}")
    print(f"True R-peaks saved to {output_dir / 'true_r_peaks.npy'}")
    return true_r_peaks


def evaluate_peak_detection(
    ecg_signal: np.ndarray,
    detected_peaks: np.ndarray,
    true_peaks: np.ndarray,
    tolerance_samples: int = 5,
) -> Dict[str, float]:
    """
    Compute performance metrics for peak detection.

    Args:
        ecg_signal (np.ndarray): The ECG signal used for peak detection.
        detected_peaks (np.ndarray): Indices of peaks detected by the algorithm.
        true_peaks (np.ndarray): Indices of the true R peaks in the signal.
        tolerance_samples (int, optional): Number of samples of tolerance for considering a
                                         detected peak as a true positive. Defaults to 5.

    Returns:
        Dict[str, float]: Dictionary containing various performance metrics:
            - 'true_positives': Number of correctly detected peaks
            - 'false_positives': Number of incorrectly detected peaks
            - 'false_negatives': Number of missed true peaks
            - 'sensitivity': True positive rate (recall)
            - 'precision': Precision of the detection
            - 'f1_score': F1 score (harmonic mean of precision and sensitivity)
            - 'detected_count': Total number of detected peaks
            - 'true_count': Total number of true peaks
            - 'mean_residual': Mean of the time differences between detected and true peaks
            - 'var_residual': Variance of the time differences
            - 'p_normality': p-value from Shapiro-Wilk test for normality of residuals
            - 'p_wilcoxon': p-value from Wilcoxon signed-rank test
            - 'residuals': List of time differences between detected and true peaks
    """
    # Initialize metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    residuals = []

    # Count true positives and false positives.
    for detected in detected_peaks:
        closest_idx = np.argmin(np.abs(true_peaks - detected))
        if abs(detected - true_peaks[closest_idx]) <= tolerance_samples:
            true_positives += 1
        else:
            false_positives += 1

    matched_true_peaks = 0
    for true_peak in true_peaks:
        if len(detected_peaks) > 0:
            closest_idx = np.argmin(np.abs(detected_peaks - true_peak))
            diff = detected_peaks[closest_idx] - true_peak
            if abs(diff) <= tolerance_samples:
                residuals.append(diff)
                matched_true_peaks += 1
    false_negatives = len(true_peaks) - matched_true_peaks

    sensitivity = true_positives / len(true_peaks) if len(true_peaks) > 0 else 0
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    f1_score = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0
    )

    if len(residuals) > 0:
        mean_res = np.mean(residuals)
        var_res = np.var(residuals)
        if np.ptp(residuals) == 0:
            p_norm = 1.0
        else:
            try:
                _, p_norm = shapiro(residuals)
            except Exception:
                p_norm = np.nan
        try:
            if len(residuals) >= 10 and np.ptp(residuals) != 0:
                _, p_wilcoxon = wilcoxon(residuals)
            else:
                p_wilcoxon = np.nan
        except Exception:
            p_wilcoxon = np.nan
    else:
        mean_res = np.nan
        var_res = np.nan
        p_norm = np.nan
        p_wilcoxon = np.nan

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "sensitivity": sensitivity,
        "precision": precision,
        "f1_score": f1_score,
        "detected_count": len(detected_peaks),
        "true_count": len(true_peaks),
        "mean_residual": mean_res,
        "var_residual": var_res,
        "p_normality": p_norm,
        "p_wilcoxon": p_wilcoxon,
        "residuals": residuals,
    }


def main() -> None:
    """
    Main function to generate and evaluate ECG signals.

    This function:
    1. Sets up the necessary directories for data storage
    2. Generates synthetic ECG signals with various SNR levels
    3. Evaluates multiple peak detection methods on both clean and noisy signals
    4. Computes performance metrics for each method and SNR level
    5. Performs statistical analysis (ANOVA) to compare methods
    6. Saves all results to a pickle file for further analysis

    The function uses the following peak detection methods:
    - nabian2018
    - pantompkins1985
    - hamilton2002
    - elgendi2010
    - engzeemod2012
    - kalidas2017
    - martinez2004
    - rodrigues2021
    - manikandan2012

    Returns:
        None
    """
    # Use the folder "synthetic_results" inside the "data" directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)

    output_dir = data_dir / "synthetic_results"
    output_dir.mkdir(exist_ok=True)
    template_path = data_dir / "ecg_noise_template.pkl"

    peaks = 10000  # Number of R-peaks for the clean ECG signal.
    prop = 0.05  # Proportion of beats to add random deviations.
    true_r_peaks = create_expanded_snr_dataset(
        output_dir, template_path, overwrite=True, num_peaks=peaks, dev_prop=prop
    )
    if true_r_peaks is None or len(true_r_peaks) == 0:
        true_r_peaks = np.load(output_dir / "true_r_peaks.npy")

    # Gather available SNR files.
    snr_files = [f for f in output_dir.glob("synthetic_ecg_snr_*.csv")]
    snr_levels = []
    processed_files = []
    for special_snr in [0.1, 0.5, 0.8]:
        filename = f"synthetic_ecg_snr_{special_snr}.csv"
        if any(f.name == filename for f in snr_files):
            snr_levels.append(special_snr)
            processed_files.append(filename)
            print(f"Found special SNR file: {filename}")
    for f in snr_files:
        if f.name in processed_files:
            continue
        snr_str = f.stem.split("_")[-1]
        try:
            if snr_str.isdigit():
                snr_levels.append(int(snr_str))
            else:
                snr_levels.append(float(snr_str))
            processed_files.append(f.name)
        except ValueError:
            print(f"Warning: Could not parse SNR value from filename: {f}")
    snr_levels = sorted(list(set(snr_levels)), reverse=True)
    print(f"Processing the following SNR levels: {snr_levels}")

    clean_df = pd.read_csv(output_dir / "synthetic_ecg_clean.csv")
    t = clean_df["time"].values
    clean_ecg = clean_df["ecg"].values
    fs = 1.0 / (t[1] - t[0])

    methods = [
        "nabian2018",
        "pantompkins1985",
        "hamilton2002",
        "elgendi2010",
        "engzeemod2012",
        "kalidas2017",
        "martinez2004",
        "rodrigues2021",
        "manikandan2012",
    ]

    results = []
    residuals_dict = {}  # Raw residuals for each (method, SNR)

    print("Evaluating peak detection methods on clean ECG...")
    for method in methods:
        try:
            print(f"  Testing method: {method}")
            detected_peaks, _ = nk.ecg_peaks(clean_ecg, sampling_rate=fs, method=method)
            peak_indices = np.where(detected_peaks["ECG_R_Peaks"])[0]
            metrics = evaluate_peak_detection(clean_ecg, peak_indices, true_r_peaks)
            metrics["method"] = method
            metrics["snr"] = "Clean"
            results.append(metrics)
            residuals_dict[(method, "Clean")] = metrics["residuals"]
        except Exception as e:
            print(f"  Error with method {method}: {e}")

    print("\nEvaluating peak detection methods on noisy ECG signals...")
    for snr in snr_levels:
        print(f"Processing SNR = {snr}")
        try:
            filepath = output_dir / f"synthetic_ecg_snr_{snr}.csv"
            if not filepath.exists():
                if snr == int(snr):
                    filepath = output_dir / f"synthetic_ecg_snr_{int(snr)}.csv"
                if not filepath.exists():
                    print(f"Cannot find file for SNR={snr}")
                    continue
            noisy_df = pd.read_csv(filepath)
            noisy_ecg = noisy_df["ecg"].values

            for method in methods:
                try:
                    print(f"  Testing method: {method} at SNR={snr}")
                    detected_peaks, _ = nk.ecg_peaks(
                        noisy_ecg, sampling_rate=fs, method=method
                    )
                    peak_indices = np.where(detected_peaks["ECG_R_Peaks"])[0]
                    metrics = evaluate_peak_detection(
                        noisy_ecg, peak_indices, true_r_peaks
                    )
                    metrics["method"] = method
                    metrics["snr"] = snr
                    results.append(metrics)
                    residuals_dict[(method, snr)] = metrics["residuals"]
                except Exception as e:
                    print(f"  Error with method {method} at SNR={snr}: {e}")
        except Exception as e:
            print(f"Error processing SNR={snr}: {e}")

    results_df = pd.DataFrame(results)

    # ANOVA: compare residuals across methods for each SNR.
    anova_results = {}
    snr_values = list(set([r for (_, r) in residuals_dict.keys()]))
    for snr in snr_values:
        group_residuals = []
        for method in methods:
            key = (method, snr)
            if key in residuals_dict:
                r_list = residuals_dict[key]
                if len(r_list) > 0:
                    group_residuals.append(r_list)
        if len(group_residuals) > 1:
            try:
                _, p_val = f_oneway(*group_residuals)
            except Exception:
                p_val = np.nan
            anova_results[snr] = p_val
        else:
            anova_results[snr] = np.nan

    # Round numeric columns to 3 decimal places.
    results_df_numeric = results_df.copy()
    numeric_cols = results_df_numeric.select_dtypes(include=[np.number]).columns
    results_df_numeric[numeric_cols] = results_df_numeric[numeric_cols].round(3)

    # Also round ANOVA results.
    anova_results_rounded = {
        snr: round(p, 3) if not pd.isna(p) else p for snr, p in anova_results.items()
    }

    # Save all the results (both DataFrame and ANOVA outcomes) in a dictionary to a pickle.
    data_dict = {
        "results_df": results_df_numeric,
        "anova_results": anova_results_rounded,
    }
    pickle_path = data_dir / "peak_detection_results.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(
        f"Peak detection evaluation complete. Results saved as pickle at: {pickle_path}"
    )


if __name__ == "__main__":
    main()
