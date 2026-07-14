import numpy as np
import math
from typing import List
from abc import ABC, abstractmethod


class PeriodicFunction(ABC):
    """
    Abstract Base Class (ABC) for defining periodic basis functions.
    """
    @abstractmethod
    def func(self, x: float) -> np.ndarray:
        """Generates the feature vector for a given point x."""
        pass
    
    @abstractmethod
    def set_dim(self, m: List[int]):
        """Sets the dimensionality of the basis functions."""
        pass
        
    @abstractmethod
    def set_pattern_period(self, p: float):
        """Configures the fundamental period."""
        pass

# --- Constants ---
HOUR_IN_SECONDS = 3600


class FourierExpansionFunction(PeriodicFunction):
    """
    Implementation of a Fourier Series expansion as a set of basis functions.
    
    This class generates a feature vector containing a constant term (a0) 
    followed by sine and cosine pairs of increasing harmonic frequencies. 
    It is used by RidgeFE models to capture complex periodic behaviors 
    in telemetry data.
    
    Basis vector format: [1.0, sin(wx), cos(wx), sin(2wx), cos(2wx), ...]
    where w is the fundamental angular frequency.
    """

    def __init__(self):
        """Initializes the FourierExpansionFunction."""
        self.m: int = 0  # Total number of features (parameters)
        self.freq: float = 0.0  # Fundamental angular frequency

    def get_trend(self, params: np.ndarray, time: float) -> float:
        """
        Calculates the predicted value for a specific timestamp.
        
        Formula: y = Σ(basis_i(time) * params_i)

        Args:
            params (np.ndarray): Learned Fourier coefficients.
            time (float): The timestamp in hours since the cycle start.

        Returns:
            float: The predicted telemetry value.
        """
        cal_values = self.func(time)
        return float(np.dot(cal_values, params))

    def func(self, x: float) -> np.ndarray:
        """
        Generates the Fourier basis vector for a given value x.

        Args:
            x (float): The input variable (usually hours since cycle reference).

        Returns:
            np.ndarray: The feature vector [1.0, sin(wx), cos(wx), ...].
        """
        values = np.zeros(self.m)
        
        if self.m > 0:
            values[0] = 1.0  # Intercept/DC term
            
            order = (self.m - 1) // 2
            for i in range(1, order + 1):
                angle = i * self.freq * x
                # Indexing matches the Java layout: [1, s1, c1, s2, c2, ...]
                values[2 * i - 1] = math.sin(angle)
                values[2 * i] = math.cos(angle)
                
        return values

    def get_values(self, time: float, params: np.ndarray) -> float:
        """
        Alias for get_trend. Calculates predicted value using dot product.
        """
        cal_values = self.func(time)
        return float(np.dot(cal_values, params))

    def set_dim(self, m: List[int]):
        """
        Sets the total parameter dimension for the Fourier expansion.

        Args:
            m (List[int]): List where the first element is the term count.
        """
        if m:
            self.m = m[0]

    def set_pattern_period(self, p: float):
        """
        Sets the cycle period and derives the fundamental angular frequency.

        Args:
            p (float): The period in seconds.
        """
        if p > 0:
            # Derived frequency assuming input 'x' in func() is in hours
            self.freq = 2 * math.pi * HOUR_IN_SECONDS / p
        else:
            self.freq = 0.0
