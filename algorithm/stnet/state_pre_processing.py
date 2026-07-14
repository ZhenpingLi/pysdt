import logging
import os
import sys
from typing import Optional

import numpy as np

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.training_set import TrainingSet
from training.preprocessing.coefficient_inf import CoefficientInf
from training.preprocessing.orbit_based_transform import MIN, MAX, MINMAX, UNIFORM, SCALE, OFFSET, NETRANGE
from training.preprocessing.min_max_coef import MinMaxCoef
from training.preprocessing.min_coef import MinCoef
from training.preprocessing.max_coef import MaxCoef
from training.preprocessing.uniform_coef import UniformCoef

# --- Constants ---
CONTEXT = "StatePreProcessing"


class StatePreProcessing:
    """
    Preprocessing component for State-based Neural Network (STNET) models.
    
    This class is responsible for normalizing telemetry data within specific 
    operational states. It applies various scaling strategies (Min, Max, MinMax, 
    Uniform) to transform raw values into a standardized range (e.g., [-0.9, 0.9]) 
    suitable for neural network training.
    """

    def __init__(self):
        """Initializes the StatePreProcessing component."""
        self.calculate_coef: Optional[CoefficientInf] = None

    def transform(self, training_set: TrainingSet, norm_type: Optional[str]):
        """
        Applies the specified normalization transformation to the training set's outputs.
        
        This method selects a coefficient calculation strategy based on `norm_type`, 
        computes the scale and offset coefficients for each pattern cycle, and 
        then applies these transformations to the raw output values. The final 
        outputs are clamped to a predefined range (NETRANGE) for stability.

        Args:
            training_set (TrainingSet): The dataset to be transformed. Modified in-place.
            norm_type (Optional[str]): The type of normalization to apply 
                ('min', 'max', 'minmax', 'uniform'). Defaults to 'minmax'.
        """
        if not training_set or training_set.raw.size == 0:
            logging.warning(f"{CONTEXT}: Empty training set provided for transformation.")
            return

        # 1. Select the appropriate coefficient calculation strategy
        if norm_type == MIN:
            self.calculate_coef = MinCoef()
        elif norm_type == MAX:
            self.calculate_coef = MaxCoef()
        elif norm_type == UNIFORM:
            self.calculate_coef = UniformCoef()
        elif norm_type == MINMAX or norm_type is None:
            self.calculate_coef = MinMaxCoef()
        else:
            logging.warning(f"{CONTEXT}: Unknown normalization type '{norm_type}'. Defaulting to MinMax.")
            self.calculate_coef = MinMaxCoef()

        # 2. Calculate Scale/Offset coefficients based on per-cycle statistics
        stat_list = training_set.stat_list
        if not stat_list:
            logging.warning(f"{CONTEXT}: Stat list is empty. Cannot perform normalization.")
            return
            
        so_list = self.calculate_coef.get_coefficient(stat_list)
        training_set.coef_list = so_list

        # 3. Apply the transformation to the raw outputs
        inputs = training_set.inputs
        raw_outputs = training_set.outputs
        
        valid_so_points = [dp for dp in so_list if dp is not None]
        if not valid_so_points:
            logging.warning(f"{CONTEXT}: No valid scale/offset coefficients. Outputs remain untransformed.")
            training_set.outputs = training_set.raw
            return

        # Vectorized application of coefficients for efficiency
        times = inputs[:, 0]
        so_times = np.array([dp.time for dp in valid_so_points])
        
        # Find the correct coefficient set for each data point
        indices = np.searchsorted(so_times, times, side='right') - 1
        indices = np.clip(indices, 0, len(valid_so_points) - 1) # Ensure indices are within bounds
        
        # Extract scales and offsets corresponding to each data point
        scales = np.array([valid_so_points[i].data[SCALE] for i in indices])
        offsets = np.array([valid_so_points[i].data[OFFSET] for i in indices])
        
        # Perform the transformation: y' = scale * (y - offset)
        transformed_values = scales * (raw_outputs - offsets)
        
        # Clamp the results to the predefined range [-NETRANGE, NETRANGE]
        final_output = np.clip(transformed_values, -NETRANGE, NETRANGE)
        
        training_set.outputs = final_output
