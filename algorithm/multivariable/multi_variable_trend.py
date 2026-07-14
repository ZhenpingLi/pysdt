import os
import sys
from typing import List, Optional

import numpy as np

from sdtdb import sdt_db

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_trend import DataTrend
import training.data_buffer

# --- Constants ---
LONGTERM = 1
MEAN_INDEX = 2


class MultiVariableTrend(DataTrend):
    """
    Predictive model implementation for multi-variable polynomial trends.
    
    This model captures the dependence of a primary telemetry point on one 
    or more independent input variables. It constructs a feature vector 
    [1.0, x1, x2, ..., xm] and uses learned coefficients to perform predictions.
    """

    def __init__(self, mnemonic_id: str):
        """
        Initializes a new MultiVariableTrend object.

        Args:
            mnemonic_id (str): Unique identifier for the telemetry mnemonic.
        """
        super().__init__(mnemonic_id)
        
        self.m: int = 0
        self.s_index: int = 0
        
        if self.algorithm:
            dim = self.algorithm.get_dimension()
            if len(dim) == 1:
                self.s_index = 0
                self.m = dim[0]
            elif len(dim) >= 2:
                self.s_index = dim[0]
                self.m = dim[1]
        depends = sdt_db.get_mnemonic_type(mnemonic_id).depends
        num_depends = 1
        if depends:
            num_depends = len(depends.split('|'))
        # Total parameters = bias (1) + number of variables (m)
        self.param_dim = self.m*num_depends + 1
        self.alg_name = "mpoly"

    def get_pattern_period(self) -> float:
        """Returns the duration of a single pattern cycle."""
        return self.pattern_period

    def get_trend_value(self, time: List[float]) -> float:
        """
        Calculates the predicted trend value for a multi-dimensional input.

        Args:
            time (List[float]): Input vector where time[0] is the timestamp 
                and subsequent elements are the independent variables.

        Returns:
            float: The predicted telemetry value.
        """
        if self.params is not None and len(time) > self.m:
            features = self.poly_function(time)
            # Predict using linear combination: y = weights \cdot features
            output_value = np.dot(features, self.params)
            return float(output_value)
        else:
            # Fallback to mean if model is uninitialized
            if self.stat_list and self.stat_list[0]:
                return float(self.stat_list[0].data[MEAN_INDEX])
            return 0.0

    def get_param_dim(self) -> int:
        """Returns the number of variables (m) plus the bias term."""
        return self.param_dim

    def poly_function(self, _input: List[float]) -> List[float]:
        """
        Transforms raw inputs into a polynomial feature vector [1.0, v1, v2, ..., vm].

        Args:
            _input (List[float]): Raw vector [time, var1, var2, ...].

        Returns:
            List[float]: The feature vector of length m + 1.
        """
        # Initialize features with bias term 1.0
        features = [1.0] * (self.m *(len(_input)-1) + 1)
        # Fill variables from input, skipping time at index 0
        limit = self.m * (len(_input) - 1)
        features[1 : 1 + limit] = _input[1 : 1 + limit]
        return features
