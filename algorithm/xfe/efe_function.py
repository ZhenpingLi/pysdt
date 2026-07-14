import numpy as np
import math
from typing import List
import sys
import os

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.ridgefe.fourier_expansion_function import PeriodicFunction

# --- Constants ---
HOUR_IN_SECONDS = 3600


class EFEFunction(PeriodicFunction):
    """
    Implementation of the Extended Fourier Expansion (EFE) basis functions.
    
    This class generates a complex basis vector that combines a standard 
    Fourier series, a linear drift term, and a time-modulated Fourier series. 
    It is used to model telemetry patterns that not only repeat periodically 
    but also evolve or drift over time.
    
    Mathematical formula:
    f(t) = C + At + Σ[a_i*sin(iωt) + b_i*cos(iωt)] + t*Σ[c_j*sin(jωt) + d_j*cos(jωt)]
    
    Basis vector layout:
    [1.0, sin(ωt), cos(ωt), ..., t, t*sin(ωt), t*cos(ωt), ...]
    """

    def __init__(self):
        """Initializes the EFEFunction with zeroed orders."""
        self.m: List[int] = [0, 0]  # [Standard Order, Modulated Order]
        self.order: int = 0
        self.freq: float = 0.0

    def get_trend(self, params: np.ndarray, time: float) -> float:
        """
        Calculates the predicted model value for a specific time and parameters.
        
        Formula: y = Σ(basis_i(time) * params_i)

        Args:
            params (np.ndarray): Learned coefficients of length 2*(m0+m1) + 2.
            time (float): The normalized timestamp in hours.

        Returns:
            float: The predicted telemetry value.
        """
        cal_values = self.func(time)
        # Prediction using dot product of basis vector and weights
        return float(np.dot(cal_values, params))

    def set_pattern_period(self, p: float):
        """
        Sets the cycle period and derives the fundamental angular frequency.

        Args:
            p (float): The period in seconds.
        """
        if p > 0:
            # Derived frequency assuming input 't' in func() is in hours
            self.freq = 2.0 * math.pi * HOUR_IN_SECONDS / p
        else:
            self.freq = 0.0

    def func(self, x: float) -> np.ndarray:
        """
        Generates the EFE basis vector for a given value x.

        Args:
            x (float): The input variable (usually hours relative to epoch).

        Returns:
            np.ndarray: The feature vector containing DC, harmonics, drift, 
                and modulated harmonics.
        """
        # Total params = 1 (constant) + 2*m[0] (harmonics) + 1 (drift) + 2*m[1] (modulated)
        num_params = 2 * self.order + 2
        values = np.zeros(num_params)
        
        # 1. Constant term (C)
        values[0] = 1.0
        
        # 2. Standard Fourier series part
        offset = 1
        for i in range(1, self.m[0] + 1):
            angle = i * self.freq * x
            values[offset] = math.sin(angle)
            values[offset + 1] = math.cos(angle)
            offset += 2
            
        # 3. Linear drift term (At)
        values[offset] = x
        offset += 1
        
        # 4. Time-modulated Fourier series part
        for i in range(1, self.m[1] + 1):
            angle = i * self.freq * x
            values[offset] = x * math.sin(angle)
            values[offset + 1] = x * math.cos(angle)
            offset += 2
            
        return values

    def set_dim(self, _m: List[int]):
        """
        Configures the harmonic orders for standard and modulated series.

        Args:
            _m (List[int]): Dimensions [m0, m1].
        """
        if _m and len(_m) >= 2:
            self.m = [_m[0], _m[1]]
            self.order = self.m[0] + self.m[1]
        else:
            self.m = [0, 0]
            self.order = 0
