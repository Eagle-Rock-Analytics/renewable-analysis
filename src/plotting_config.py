"""Consistent visualization settings for renewable analysis notebooks.

This module provides standardized color palettes, markers, and other plotting
configurations to ensure consistency across all analysis notebooks.
"""

# Model colors from lipari palette for consistency across plots
model_colors = {
    "ec-earth3": "#13385A",  # lipari-29 (deep blue)
    "miroc6": "#E37861",  # lipari-169 (orange-red)
    "mpi-esm1-2-hr": "#785F72",  # lipari-97 (purple-ish mid tone)
    "taiesm1": "#CD685F",  # lipari-153 (coral red)
}

# GWL colors for reference (0.8°C) and future (2.0°C) periods
# These are lighter versions suitable for fill/shading
gwl_colors = {
    0.8: "#879FF5",  # Light blue for reference period (lipari-based)
    2.0: "#EC9B97",  # Light coral for future period (lipari-based)
    3.0: "#E37861",  # Orange-red for higher future GWL (lipari-169)
}

# Marker styles for each model (useful for scatter plots)
model_markers = {
    "ec-earth3": "o",  # circle
    "miroc6": "s",  # square
    "mpi-esm1-2-hr": "^",  # triangle up
    "taiesm1": "D",  # diamond
}

# Full lipari 10-color palette (for extended use cases)
lipari_10 = [
    "#031326",  # lipari-1 (darkest)
    "#13385A",  # lipari-29
    "#47587A",  # lipari-58
    "#6B5F76",  # lipari-86
    "#8E616C",  # lipari-114
    "#BC6461",  # lipari-143
    "#E57B62",  # lipari-171
    "#E7A279",  # lipari-199
    "#E9C99F",  # lipari-228
    "#FDF5DA",  # lipari-256 (lightest)
]

# Model order for consistent legend/plot ordering
model_order = ["ec-earth3", "miroc6", "mpi-esm1-2-hr", "taiesm1"]
