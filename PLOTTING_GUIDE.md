# Plotting Configuration Guide

This guide explains how to use the standardized plotting configuration in the renewable-analysis project.

## Quick Start

### 1. Import plotting configuration

```python
from renewable_data_load import *
from plotting_config import model_colors, gwl_colors, model_markers
import matplotlib.pyplot as plt

# Load the style sheet
plt.style.use('../../renewable_analysis.mplstyle')  # adjust path as needed
```

### 2. Use model colors explicitly

```python
# Plot each model with consistent colors
for model in ['ec-earth3', 'miroc6', 'mpi-esm1-2-hr', 'taiesm1']:
    data = get_data_for_model(model)
    plt.plot(data, color=model_colors[model], label=model)
```

### 3. Use GWL colors for shading

```python
# Reference period (0.8°C GWL)
plt.fill_between(x, y_min, y_max, color=gwl_colors[0.8], alpha=0.2, label='GWL 0.8°C')

# Future period (2.0°C GWL)
plt.fill_between(x, y_min, y_max, color=gwl_colors[2.0], alpha=0.2, label='GWL 2.0°C')
```

### 4. Use markers for scatter plots

```python
for model in model_colors.keys():
    plt.scatter(x, y,
                color=model_colors[model],
                marker=model_markers[model],
                label=model,
                s=100)
```

## Configuration Details

### Model Colors (Lipari Palette)

- **ec-earth3**: `#13385A` (lipari-29, deep blue)
- **miroc6**: `#E37861` (lipari-169, orange-red)
- **mpi-esm1-2-hr**: `#785F72` (lipari-97, purple-ish mid tone)
- **taiesm1**: `#CD685F` (lipari-153, coral red)

### GWL Colors

- **0.8°C** (reference): `#879FF5` (light blue)
- **2.0°C** (future): `#EC9B97` (light coral)

### Model Markers

- **ec-earth3**: `'o'` (circle)
- **miroc6**: `'s'` (square)
- **mpi-esm1-2-hr**: `'^'` (triangle up)
- **taiesm1**: `'D'` (diamond)

## Style Sheet Settings

The `renewable_analysis.mplstyle` file includes:

- **Figure size**: 18 x 12 inches (default)
- **DPI**: 100 (display), 400 (saved figures)
- **Font sizes**: Title=18, Axes Title=16, Labels=14, Ticks=12, Legend=12
- **Grid**: Enabled with alpha=0.3
- **Color cycle**: Uses the four model colors in order

## Files

- `src/plotting_config.py` - Color and marker definitions
- `renewable_analysis.mplstyle` - Matplotlib style sheet
- Individual notebooks import from `plotting_config` as needed

## Best Practices

1. **Always use explicit model colors** when models may appear in different orders across subplots
2. **Load the style sheet** at the beginning of each notebook for consistent formatting
3. **Use `model_markers`** for scatter plots to distinguish models when color isn't sufficient
4. **Use `gwl_colors`** with `alpha` parameter for shaded regions (e.g., model spread)

## Example: Complete Setup

```python
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from renewable_data_load import *
from plotting_config import model_colors, gwl_colors, model_markers

# Load style sheet
plt.style.use('../../renewable_analysis.mplstyle')

# Create figure
fig, ax = plt.subplots()

# Plot with explicit colors
for model in model_colors.keys():
    data = load_model_data(model)
    ax.plot(data, color=model_colors[model], label=model, linewidth=2)

ax.legend()
ax.set_title('Model Comparison')
plt.savefig('figures/comparison.png')  # Saves at 400 dpi automatically
```

## Color Palette Reference

The `plotting_config` module also includes `lipari_10`, a full 10-color palette from the Lipari colormap for extended use cases where you need more than 4 colors.
