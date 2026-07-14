import math
import numpy as np
from config.sdt_constants import DAY_IN_SECONDS

# --- Global State ---
param_size: int = 0
"""The total number of coefficients in the Fourier expansion (DC + 2*Order)."""

frequency: float = 0.0
"""The fundamental angular frequency for the periodic basis functions."""


def func(x: float) -> np.ndarray:
    """
    Generates the Fourier basis vector for a normalized timestamp x.
    
    The vector follows the format: [1.0, sin(wx), cos(wx), sin(2wx), cos(2wx), ...]
    where 'w' is the angular frequency configured for the current model.

    Args:
        x (float): The normalized timestamp (usually hours since cycle start).

    Returns:
        np.ndarray: The resulting feature vector of length 'param_size'.
    """
    global param_size, frequency
    values = np.zeros(param_size)
    
    if param_size > 0:
        # Intercept / DC term
        values[0] = 1.0
        
        # Harmonic terms: calculate sine and cosine pairs
        order = (param_size - 1) // 2
        for i in range(1, order + 1):
            angle = i * frequency * x
            # Alternating sin/cos pairs
            values[2 * i - 1] = np.sin(angle)
            values[2 * i] = np.cos(angle)
            
    return values

def get_values(time: float, params: np.ndarray) -> float:
    """
    Calculates the model output value for a specific timestamp and weight set.
    
    It generates the Fourier basis functions for the provided time and 
    calculates the prediction using the dot product of basis and coefficients.

    Args:
        time (float): The normalized timestamp.
        params (np.ndarray): The learned Fourier coefficients.

    Returns:
        float: The predicted telemetry value.
    """
    cal_values = func(time)
    # y = Σ(basis_i * weights_i)
    fit_value = np.dot(cal_values, params)
    return float(fit_value)

def set_dim(m: int):
    """
    Sets the total dimensionality (number of terms) for the Fourier expansion.

    Args:
        m (int): The parameter count.
    """
    global param_size
    param_size = m

def set_period(p: float):
    """
    Sets the cycle duration and derives the fundamental angular frequency.
    
    The frequency is calculated assuming the input 'x' in func() is in days.

    Args:
        p (float): The period duration in seconds.
    """
    global frequency
    if p > 0:
        # Angular frequency formula: w = 2*pi / T
        # T (in days) = p / DAY_IN_SECONDS
        frequency = 2 * math.pi * DAY_IN_SECONDS / p
    else:
        frequency = 0.0

def set_pattern_period(p: float):
    """
    Alias for set_period. Configures the fundamental periodic duration.
    """
    set_period(p)
