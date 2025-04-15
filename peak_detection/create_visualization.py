#!/usr/bin/env python
"""
Module for creating visualizations of peak detection results.

This module provides functions to create and save heatmap visualizations
of various metrics from peak detection results.
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set global style options
sns.set_context("notebook", rc={"axes.labelweight": "bold"})
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def create_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    colorbar_label: str,
    min_val: float = 0,
    break_out: float = 0.5,
    max_val: float = 1,
    cmap_name: str = "coolwarm",
    filename: str = "not_named",
    output_dir: Union[str, Path] = None,
    dpi: int = 900,
) -> None:
    """
    Create and save a heatmap visualization of a given metric.

    Args:
        df: DataFrame containing the results.
        metric: Name of the column in df to visualize.
        title: Title of the heatmap.
        colorbar_label: Label for the colorbar.
        min_val: Minimum value for the color scale.
        break_out: Center value for the color scale.
        max_val: Maximum value for the color scale.
        cmap_name: Name of the colormap to use.
        filename: Output filename (e.g., 'f1_score_heatmap.png').
        output_dir: Directory where the image will be saved.
        dpi: Resolution of the saved image.

    Returns:
        None
    """
    if output_dir is None:
        raise ValueError("output_dir must be specified")

    font_size = 15
    plt.figure(figsize=(15, 7))

    # Create pivot table for heatmap
    pivot_data = df.pivot_table(index="method", columns="snr", values=metric)

    # Create heatmap
    ax = sns.heatmap(
        pivot_data,
        annot=True,
        cmap=cmap_name,
        vmin=min_val,
        vmax=max_val,
        center=break_out,
        fmt=".3f",
        annot_kws={"fontsize": font_size},
        cbar_kws={"label": colorbar_label},
    )

    # Customize appearance
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel("SNR", fontsize=font_size)
    ax.set_ylabel("", fontsize=1)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=font_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size, fontweight="bold")

    # Save figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"{title} saved to: {output_path}")


def main() -> None:
    """
    Main function to create visualizations from peak detection results.

    Loads results from a pickle file and creates various heatmap visualizations
    for different metrics.
    """
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    pickle_path = os.path.join(data_dir, "peak_detection_results.pkl")
    output_dir = os.path.join(data_dir, "visualization_results")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    with open(pickle_path, "rb") as f:
        data_dict = pickle.load(f)
        results_df = data_dict["results_df"]

    # Sort results
    results_df["snr_num"] = pd.to_numeric(results_df["snr"], errors="coerce")
    results_df = results_df.sort_values(["method", "snr_num"], ascending=[True, False])

    # Create visualizations
    visualizations = [
        {
            "metric": "f1_score",
            "title": "F1 Score by Method and SNR",
            "colorbar_label": "F1 Score (AU)",
            "cmap_name": "coolwarm_r",
            "filename": "f1_score_heatmap.png",
        },
        {
            "metric": "sensitivity",
            "title": "Sensitivity by Method and SNR",
            "colorbar_label": "Sensitivity (AU)",
            "cmap_name": "coolwarm_r",
            "filename": "sensitivity_heatmap.png",
        },
        {
            "metric": "precision",
            "title": "Precision by Method and SNR",
            "colorbar_label": "Precision (AU)",
            "cmap_name": "coolwarm_r",
            "filename": "precision_heatmap.png",
        },
        {
            "metric": "mean_residual",
            "title": "Mean Residuals by Method and SNR",
            "colorbar_label": "Mean Residual (ms)",
            "min_val": -2,
            "break_out": 0,
            "max_val": 2,
            "cmap_name": "coolwarm",
            "filename": "mean_residuals_heatmap.png",
        },
        {
            "metric": "var_residual",
            "title": "Variance of Residuals by Method and SNR",
            "colorbar_label": "Variance of Residual (ms²)",
            "min_val": 0,
            "break_out": 20,
            "max_val": 40,
            "cmap_name": "coolwarm",
            "filename": "var_residuals_heatmap.png",
        },
        {
            "metric": "p_normality",
            "title": "Normality Test p-values by Method and SNR",
            "colorbar_label": "p-value (Normality)",
            "cmap_name": "coolwarm",
            "filename": "p_normality_heatmap.png",
        },
        {
            "metric": "p_wilcoxon",
            "title": "Wilcoxon Test p-values by Method and SNR",
            "colorbar_label": "p-value (Wilcoxon)",
            "cmap_name": "coolwarm",
            "filename": "p_wilcoxon_heatmap.png",
        },
    ]

    # Generate all visualizations
    for viz in visualizations:
        print(f"Creating {viz['title']}...")
        create_heatmap(results_df, output_dir=output_dir, **viz)


if __name__ == "__main__":
    main()
