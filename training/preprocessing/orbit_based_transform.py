import logging
import os
import sys
from typing import List, Optional

import numpy as np

from config.sdt_constants import SCALE, OFFSET

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_point import DataPoint
from algorithm.algorithm_def import AlgorithmDef
from training.training_set import TrainingSet
from training.data_buffer import get_default_model_time
from .min_coef import MinCoef
from .max_coef import MaxCoef
from .min_max_coef import MinMaxCoef
from .uniform_coef import UniformCoef


# Constants
CONTEXT = "OBTRANSFORM"
NETRANGE = 0.9
MAX = "max"
MIN = "min"
MINMAX = "minmax"
SHORTTERM = 0
WARNING = "WARNING"
NORMTYPE = "normtype"
UNIFORM = "uniform"


from bisect import bisect_right

def get_so_data(so_list: List[DataPoint], time: float) -> Optional[np.ndarray]:
    """
    Retrieves the scale and offset coefficients valid for a specific timestamp.
    
    Uses binary search (bisect_right) to find the correct time-tagged coefficient 
    record within a sorted list.

    Args:
        so_list (List[DataPoint]): A list of DataPoints containing coefficients.
        time (float): The timestamp to query.

    Returns:
        Optional[np.ndarray]: The coefficient data array [offset, scale], or 
            None if the list is empty.
    """
    if not so_list:
        return None

    # Finds the insertion point to identify the relevant coefficient segment
    idx = bisect_right(so_list, time, key=lambda x: x.time)

    if idx == 0:
        return so_list[0].data

    if idx >= len(so_list):
        return so_list[-1].data

    return so_list[idx].data



class OrbitBasedTransform:
    """
    Normalization engine for telemetry data aligned with orbital patterns.
    
    This class performs per-cycle (per-orbit) normalization of training data. 
    It calculates dynamic scale and offset coefficients based on the statistics 
    of each pattern cycle, transforming the raw values into a normalized range 
    suitable for neural network training.
    """

    def __init__(self, algorithm: AlgorithmDef, mnemonic_id: str):
        """
        Initializes the OrbitBasedTransform component.

        Args:
            algorithm (AlgorithmDef): The algorithm definition used to determine 
                the normalization strategy (e.g., MinMax, Uniform).
            mnemonic_id (str): The identifier for the mnemonic.
        """
        norm_type = algorithm.get_attribute(NORMTYPE)
        model_time = get_default_model_time(algorithm)
        self.pattern_period = model_time.get_model_period() if model_time else 0.0

        self.so_coefficients: Optional[np.ndarray] = None
        
        # Select strategy based on the 'normtype' attribute
        if norm_type == MIN:
            self.calculate_coef = MinCoef()
        elif norm_type == MAX:
            self.calculate_coef = MaxCoef()
        elif norm_type == MINMAX or norm_type is None:
            self.calculate_coef = MinMaxCoef()
        elif norm_type == UNIFORM:
            self.calculate_coef = UniformCoef()
        else:
            logging.warning(f"{CONTEXT}: {norm_type} is not defined; using MinMax as default.")
            self.calculate_coef = MinMaxCoef()

    def set_so_coef(self, coefs: np.ndarray):
        """
        Sets a fixed, static set of scale and offset coefficients.
        If set, these will override the dynamic per-cycle calculation.

        Args:
            coefs (np.ndarray): Array [offset, scale].
        """
        self.so_coefficients = coefs

    def transform(self, training_set: TrainingSet, pattern_offset: int):
        """
        Executes the normalization transformation on the TrainingSet.
        
        It generates a list of scale/offset coefficients for every pattern 
        cycle and applies them to each raw data point. Values are capped at 
        a predefined range (e.g., ±0.9) to ensure stability in gradient-based 
        learning.

        Args:
            training_set (TrainingSet): The dataset to transform.
            pattern_offset (int): Starting index for pattern cycles.
        """
        inputs = training_set.inputs
        stat_list = training_set.stat_list

        # Step 1: Generate or retrieve the scale/offset list
        if self.so_coefficients is None:
            so_list = self.calculate_coef.get_coefficient(stat_list)
        else:
            so_list = []
            for stat in stat_list:
                if stat:
                    so_list.append(DataPoint(time=stat.time, data=self.so_coefficients))
                else:
                    so_list.append(None)
                    
        training_set.coef_list = so_list

        raw = training_set.raw
        times = inputs[:, 0]
        
        valid_so_points = [dp for dp in so_list if dp is not None]
        if not valid_so_points:
            return
            
        # Step 2: Apply the transformation to each data point
        outputs = []
        for index in range(len(raw)):
            so_data = get_so_data(so_list, times[index])
            # Transformation: value = scale * (raw - offset)
            value = so_data[SCALE] * (raw[index] - so_data[OFFSET])
            
            # Capping for stability
            if abs(value) > NETRANGE:
                value = NETRANGE * np.sign(value)
            outputs.append(value)

        training_set.outputs = np.array(outputs, dtype=np.float32)
