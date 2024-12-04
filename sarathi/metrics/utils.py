from typing import Dict, List

import numpy as np


def calculate_percentile_values(
    data: List[float], percentiles: List[float] = [0, 25, 50, 75, 90, 95, 99, 100]
) -> Dict[float, float]:
    """
    Calculate the percentile values for the given data.

    Args:
        data: List of float values.
        percentiles: List of percentiles to calculate.

    Returns:
        Dictionary containing the percentile values.
    """
    percentile_values = {}
    for percentile in percentiles:
        percentile_values[percentile] = np.percentile(data, percentile)
    return percentile_values
