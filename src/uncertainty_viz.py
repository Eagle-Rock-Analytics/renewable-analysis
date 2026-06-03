"""
uncertainty_viz.py
------------------
Visualization tools for communicating climate uncertainty in grid-relevant outcomes.

This module provides functions to visualize ensemble spread, model agreement,
and IPCC-style confidence levels across the 4-model climate ensemble used in
renewable energy and grid stress analysis.

IPCC Confidence Mapping
------------------------
The functions follow IPCC AR6 guidance for communicating uncertainty:
- 4/4 models agree (same sign/direction): "Very likely" (>90% confidence)
- 3/4 models agree: "Likely" (>66% confidence)
- 2/4 models agree or high variance: "Medium confidence" (~50%)
- <2/4 models agree: "Low confidence"

Note: With only 4 models, these mappings are conservative. Standard IPCC
assessments use 40+ CMIP6 models. Interpret "very likely" as "all available
models agree" rather than 90% probability.

Example Usage
-------------
>>> from src.uncertainty_viz import plot_ensemble_spread, plot_ipcc_confidence_bars
>>> from src.coincident_analysis import load_mask_dataset, coincident_event_analysis
>>>
>>> # Load pre-computed coincident drought statistics
>>> results = coincident_event_analysis(masks, gwls=[0.8, 2.0])
>>>
>>> # Visualize ensemble spread for k=2 (both PV and wind in drought)
>>> plot_ensemble_spread(
...     results['counts'],
...     y_col='days_per_year',
...     title='Coincident PV+Wind Drought Frequency'
... )
>>>
>>> # Show model agreement with IPCC confidence
>>> plot_ipcc_confidence_bars(
...     results['ensemble'],
...     metric='days_per_year',
...     gwl_comparison=[0.8, 2.0]
... )
"""

from __future__ import annotations

from typing import Literal, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from plotting_config import gwl_colors, model_colors, model_markers, model_order

# ---------------------------------------------------------------------------
# IPCC Confidence Level Mapping
# ---------------------------------------------------------------------------


def calculate_model_agreement(
    values: list[float] | np.ndarray, threshold: float = 0.0, sign_check: bool = True
) -> tuple[int, str]:
    """
    Determine how many models agree on a change direction and assign IPCC confidence.

    Parameters
    ----------
    values : list or array
        Per-model values (e.g., change in drought days from baseline to future)
    threshold : float, default=0.0
        Value above/below which we consider a meaningful change
    sign_check : bool, default=True
        If True, check sign agreement (positive or negative change).
        If False, just count values exceeding absolute threshold.

    Returns
    -------
    n_agree : int
        Number of models agreeing (0-4)
    confidence : str
        IPCC confidence level: 'Very likely', 'Likely', 'Medium confidence', 'Low confidence'
    """
    values = np.asarray(values)
    n_models = len(values)

    if sign_check:
        # Count models with same sign of change
        n_positive = np.sum(values > threshold)
        n_negative = np.sum(values < -threshold)
        n_agree = max(n_positive, n_negative)
    else:
        # Just count models exceeding absolute threshold
        n_agree = np.sum(np.abs(values) > threshold)

    # Map to IPCC confidence levels
    if n_agree == n_models:
        confidence = "Very likely"  # All models agree (>90%)
    elif n_agree >= 0.75 * n_models:  # 3/4 models
        confidence = "Likely"  # (>66%)
    elif n_agree >= 0.5 * n_models:  # 2/4 models
        confidence = "Medium confidence"  # (~50%)
    else:
        confidence = "Low confidence"  # <50%

    return int(n_agree), confidence


def ipcc_confidence_color(confidence: str) -> str:
    """
    Return color for IPCC confidence level visualization.

    Parameters
    ----------
    confidence : str
        One of: 'Very likely', 'Likely', 'Medium confidence', 'Low confidence'

    Returns
    -------
    str
        Hex color code
    """
    color_map = {
        "Very likely": "#2166ac",  # Dark blue - high confidence
        "Likely": "#4393c3",  # Medium blue
        "Medium confidence": "#d1e5f0",  # Light blue
        "Low confidence": "#f7f7f7",  # Very light gray
    }
    return color_map.get(confidence, "#cccccc")


# ---------------------------------------------------------------------------
# Ensemble Spread Visualizations
# ---------------------------------------------------------------------------


def plot_ensemble_spread(
    df: pd.DataFrame,
    x_col: str = "gwl",
    y_col: str = "days_per_year",
    hue_col: str = "simulation",
    k_filter: Optional[int] = None,
    season_filter: Optional[str] = None,
    title: str = "Ensemble Spread Across Climate Models",
    ylabel: Optional[str] = None,
    xlabel: str = "Global Warming Level (°C)",
    figsize: tuple[float, float] = (12, 7),
    show_ensemble_mean: bool = True,
    show_model_spread: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Create a spaghetti plot showing individual model trajectories with optional ensemble statistics.

    Visualizes uncertainty by plotting each climate model as a separate line, with optional
    shading to show the full range (min-max) of the ensemble.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns for x_col, y_col, and hue_col (typically from coincident_analysis results)
    x_col : str
        Column name for x-axis (typically 'gwl' or 'time')
    y_col : str
        Column name for y-axis (e.g., 'days_per_year')
    hue_col : str
        Column name for grouping individual traces (typically 'simulation')
    k_filter : int, optional
        If provided, filter to only k=k_filter rows (for coincident drought counts)
    season_filter : str, optional
        If provided, filter to specific season ('JFM', 'AMJ', 'JAS', 'OND')
    title : str
        Plot title
    ylabel : str, optional
        Y-axis label. If None, uses y_col name.
    xlabel : str
        X-axis label
    figsize : tuple
        Figure size (width, height) in inches
    show_ensemble_mean : bool
        If True, overlay ensemble mean as thick black dashed line
    show_model_spread : bool
        If True, add shaded region showing min-max spread
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure.

    Returns
    -------
    plt.Axes
        The axes object with the plot
    """
    # Filter data if requested
    data = df.copy()
    if k_filter is not None:
        data = data[data["k"] == k_filter]
    if season_filter is not None:
        data = data[data.get("season", None) == season_filter]

    if data.empty:
        raise ValueError("No data remaining after filtering")

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot individual model trajectories
    for model in model_order:
        if model not in data[hue_col].values:
            continue
        model_data = data[data[hue_col] == model].sort_values(x_col)
        ax.plot(
            model_data[x_col],
            model_data[y_col],
            color=model_colors[model],
            marker=model_markers[model],
            markersize=8,
            linewidth=2,
            label=model.upper(),
            alpha=0.8,
        )

    # Add ensemble statistics
    if show_ensemble_mean or show_model_spread:
        ensemble_stats = data.groupby(x_col)[y_col].agg(["mean", "min", "max"])

        if show_model_spread:
            ax.fill_between(
                ensemble_stats.index,
                ensemble_stats["min"],
                ensemble_stats["max"],
                color="gray",
                alpha=0.2,
                label="Model spread (min-max)",
            )

        if show_ensemble_mean:
            ax.plot(
                ensemble_stats.index,
                ensemble_stats["mean"],
                color="black",
                linewidth=3,
                linestyle="--",
                label="Ensemble mean",
                zorder=10,
            )

    # Styling
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(
        ylabel if ylabel else y_col.replace("_", " ").title(),
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def plot_ensemble_violin(
    df: pd.DataFrame,
    x_col: str = "gwl",
    y_col: str = "days_per_year",
    hue_col: str = "simulation",
    k_filter: Optional[int] = None,
    title: str = "Distribution of Model Projections",
    ylabel: Optional[str] = None,
    xlabel: str = "Global Warming Level (°C)",
    figsize: tuple[float, float] = (10, 7),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Create violin plots showing the distribution of model values at each x-axis point.

    Useful for visualizing uncertainty when multiple GWLs or scenarios are compared.

    Parameters
    ----------
    df : pd.DataFrame
        Data with x_col, y_col, and hue_col columns
    x_col : str
        Column for x-axis grouping (typically 'gwl')
    y_col : str
        Column with values to plot
    hue_col : str
        Column identifying individual models (for counting in distribution)
    k_filter : int, optional
        Filter to specific k-count value
    title : str
        Plot title
    ylabel : str, optional
        Y-axis label
    xlabel : str
        X-axis label
    figsize : tuple
        Figure size
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Axes
        The axes object
    """
    import seaborn as sns

    # Filter data
    data = df.copy()
    if k_filter is not None:
        data = data[data["k"] == k_filter]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create violin plot
    sns.violinplot(
        data=data, x=x_col, y=y_col, ax=ax, inner="box", palette="Set2", alpha=0.7
    )

    # Overlay individual model points
    for model in model_order:
        if model not in data[hue_col].values:
            continue
        model_data = data[data[hue_col] == model]
        ax.scatter(
            model_data[x_col],
            model_data[y_col],
            color=model_colors[model],
            marker=model_markers[model],
            s=100,
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
            label=model.upper(),
        )

    # Styling
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(
        ylabel if ylabel else y_col.replace("_", " ").title(),
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="best", framealpha=0.9, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


# ---------------------------------------------------------------------------
# IPCC Confidence Visualizations
# ---------------------------------------------------------------------------


def plot_ipcc_confidence_bars(
    df: pd.DataFrame,
    metric: str = "days_per_year",
    gwl_comparison: list[float] = [0.8, 2.0],
    k_filter: Optional[int] = None,
    title: str = "Change in Grid Stress Events with IPCC Confidence",
    ylabel: str = "Change (days/year)",
    figsize: tuple[float, float] = (10, 7),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Create bar chart showing change between two warming levels with IPCC confidence annotations.

    Calculates the change (future - baseline) for each model, determines model agreement,
    and colors bars according to IPCC confidence levels.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns: 'simulation', 'gwl', and metric column
    metric : str
        Column name for the metric to analyze (e.g., 'days_per_year')
    gwl_comparison : list of two floats
        [baseline_gwl, future_gwl] to compare (e.g., [0.8, 2.0])
    k_filter : int, optional
        Filter to specific k-count
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Axes
        The axes object
    """
    data = df.copy()
    if k_filter is not None:
        data = data[data["k"] == k_filter]

    # Calculate change for each model
    baseline_gwl, future_gwl = gwl_comparison
    baseline = data[data["gwl"] == baseline_gwl].set_index("simulation")[metric]
    future = data[data["gwl"] == future_gwl].set_index("simulation")[metric]

    changes = future - baseline
    changes = changes.reindex(model_order)

    # Determine model agreement and confidence
    n_agree, confidence = calculate_model_agreement(
        changes.values, threshold=0.0, sign_check=True
    )
    color = ipcc_confidence_color(confidence)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot bars for each model
    x_pos = np.arange(len(changes))
    bars = ax.bar(
        x_pos,
        changes.values,
        color=[model_colors[m] for m in changes.index],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add ensemble mean bar
    mean_change = changes.mean()
    ax.axhline(
        mean_change,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Ensemble mean: {mean_change:.1f}",
        zorder=5,
    )

    # Add confidence level text box
    textstr = f"Model Agreement: {n_agree}/4\nIPCC Confidence: {confidence}"
    props = dict(
        boxstyle="round", facecolor=color, alpha=0.8, edgecolor="black", linewidth=2
    )
    ax.text(
        0.98,
        0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        fontweight="bold",
    )

    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [m.upper() for m in changes.index], fontsize=11, fontweight="bold"
    )
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(
        title + f"\n({baseline_gwl}°C → {future_gwl}°C warming)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.legend(loc="upper left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def create_model_agreement_matrix(
    df: pd.DataFrame,
    metrics: list[str],
    gwl_comparison: list[float] = [0.8, 2.0],
    k_filter: Optional[int] = None,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Create a summary table showing model agreement and IPCC confidence for multiple metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'simulation', 'gwl', and metric columns
    metrics : list of str
        List of column names to analyze (e.g., ['days_per_year', 'demand_days_per_year'])
    gwl_comparison : list of two floats
        [baseline_gwl, future_gwl]
    k_filter : int, optional
        Filter to specific k-count
    threshold : float
        Threshold for determining meaningful change

    Returns
    -------
    pd.DataFrame
        Summary table with columns: metric, mean_change, std_change, n_models_agree, confidence
    """
    data = df.copy()
    if k_filter is not None:
        data = data[data["k"] == k_filter]

    baseline_gwl, future_gwl = gwl_comparison
    results = []

    for metric in metrics:
        baseline = data[data["gwl"] == baseline_gwl].set_index("simulation")[metric]
        future = data[data["gwl"] == future_gwl].set_index("simulation")[metric]
        changes = future - baseline

        n_agree, confidence = calculate_model_agreement(
            changes.values, threshold=threshold
        )

        results.append(
            {
                "Metric": metric.replace("_", " ").title(),
                "Mean Change": changes.mean(),
                "Std Dev": changes.std(),
                "Min": changes.min(),
                "Max": changes.max(),
                "Models Agreeing": f"{n_agree}/4",
                "IPCC Confidence": confidence,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Uncertainty Summary Tables
# ---------------------------------------------------------------------------


def create_uncertainty_summary_table(
    results: dict,
    gwls: list[float] = [0.8, 2.0],
    k_values: Optional[list[int]] = None,
    metric: str = "days_per_year",
) -> pd.DataFrame:
    """
    Create formatted summary table of ensemble statistics with IPCC-style uncertainty language.

    Parameters
    ----------
    results : dict
        Results dictionary from coincident_event_analysis
    gwls : list of float
        GWLs to include in summary
    k_values : list of int, optional
        Specific k-counts to include. If None, includes all.
    metric : str
        Which metric to summarize

    Returns
    -------
    pd.DataFrame
        Formatted summary with ensemble mean, range, and confidence statements
    """
    counts_df = results["counts"]

    if k_values is not None:
        counts_df = counts_df[counts_df["k"].isin(k_values)]

    summary_rows = []

    for gwl in gwls:
        gwl_data = counts_df[counts_df["gwl"] == gwl]

        for k in sorted(gwl_data["k"].unique()):
            k_data = gwl_data[gwl_data["k"] == k]
            values = k_data[metric].values

            mean_val = values.mean()
            min_val = values.min()
            max_val = values.max()
            std_val = values.std()

            # Determine confidence based on coefficient of variation
            cv = std_val / mean_val if mean_val != 0 else np.inf
            if cv < 0.15:  # Low spread
                confidence = "High agreement"
            elif cv < 0.30:  # Moderate spread
                confidence = "Moderate agreement"
            else:  # High spread
                confidence = "Low agreement"

            summary_rows.append(
                {
                    "GWL (°C)": gwl,
                    "k (resources in drought)": k,
                    "Ensemble Mean": f"{mean_val:.1f}",
                    "Range": f"{min_val:.1f} – {max_val:.1f}",
                    "Std Dev": f"{std_val:.1f}",
                    "Model Agreement": confidence,
                }
            )

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Spatial Agreement Visualization (placeholder for future grid-level analysis)
# ---------------------------------------------------------------------------


def plot_model_agreement_map(
    ds: xr.Dataset,
    variable: str,
    gwl_comparison: list[float] = [0.8, 2.0],
    agreement_threshold: float = 0.0,
    title: str = "Spatial Pattern of Model Agreement",
    cmap: str = "RdYlBu_r",
    figsize: tuple[float, float] = (14, 8),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create spatial map showing where models agree on direction of change.

    NOTE: This function is a placeholder for future grid-level spatial analysis.
    Current analysis focuses on regional aggregates. Implementation requires
    gridded (x, y) data with simulation dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dims (simulation, time, y, x) and variable
    variable : str
        Variable name to analyze
    gwl_comparison : list of two floats
        [baseline_gwl, future_gwl]
    agreement_threshold : float
        Threshold for determining significant change
    title : str
        Plot title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray
        Array of axes objects [change_map, agreement_map]

    Raises
    ------
    NotImplementedError
        This function requires gridded spatial data. Current regional analysis
        uses spatial aggregates. Implement when grid-level uncertainty analysis needed.
    """
    raise NotImplementedError(
        "Spatial agreement mapping requires gridded (x, y) data with simulation dimension. "
        "Current analysis uses regional aggregates. To implement: "
        "(1) Load grid-level mask data without regional aggregation, "
        "(2) Compute per-model changes at each grid cell, "
        "(3) Count models agreeing on sign of change at each location, "
        "(4) Plot using cartopy with proper projection."
    )
