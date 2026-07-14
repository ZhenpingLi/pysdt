import os
import sys
from typing import Optional, List

import numpy as np

from training.preprocessing import data_stat

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.default.default_training_worker import DefaultTrainingWorker
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_trend import DataTrend
from training.training_set import TrainingSet
from training.preprocessing.data_stat import DataStat
from algorithm.discrete.value_counter import get_value_counts

# Constants
CONTEXT = "DiscreteTrainingWorker"
WARNING = 0
MEAN_INDEX = 2


class DiscreteTrainingWorker(DefaultTrainingWorker):
    """
    Concrete TrainingWorker implementation for quantized or discrete telemetry.
    
    This worker is designed for telemetry points that jump between distinct 
    levels (e.g., states or raw counts). It calculates a minimum significant 
    delta between levels to prevent false positive outlier detections that 
    could be caused by simple quantization noise.
    
    It inherits baseline statistical modeling from DefaultTrainingWorker but 
    overrides sigma calculation to enforce the floor defined by the delta.
    """

    def __init__(self):
        """Initializes the DiscreteTrainingWorker."""
        super().__init__()
        self.delta: float = 0.0
        self.sscale: float = 1.5
        self.bound: float = 0.0

    def set_config(self, node: MnemonicNode):
        """
        Configures the worker using algorithm attributes.

        Args:
            node (MnemonicNode): The tree node to be trained.
        """
        super().set_config(node)
        
        s_scale_string = self.algorithm.get_attribute("sscale")
        if s_scale_string:
            self.sscale = float(s_scale_string)
            
        b_string = self.algorithm.get_attribute("bound")
        if b_string:
            self.bound = float(b_string)
        else:
            self.bound = 0.0
            
        self.delta = 0.0

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Executes the training logic for discrete data.
        
        It first identifies the characteristic delta between common value 
        levels and then calculates the session mean.

        Args:
            training_set (TrainingSet): Data points to train on.
            current_trend (DataTrend): The model object to update.
            input_trend (Optional[DataTrend]): The baseline model.
        """
        self._find_delta(training_set, current_trend.get_limits()[WARNING])

        self.stat_point = data_stat.cal_stat(training_set, self.ref_time, None)
        
        params = np.zeros(1)
        if self.stat_point:
            params[0] = self.stat_point.data[MEAN_INDEX]
            
        current_trend.set_params(params)
        current_trend.set_trended(True)

    def _find_delta(self, training_set: TrainingSet, limit: float):
        """
        Identifies the typical jump distance between discrete value levels.
        
        It analyzes the frequency of unique values in the dataset and determines 
         the spread between the most frequent points. This spread is used to 
         set a conservate floor for outlier detection.

        Args:
            training_set (TrainingSet): Source data.
            limit (float): The warning limit multiplier used for scaling the delta.
        """
        outputs = training_set.outputs
        
        # Use helper to group and count unique values
        value_counters = get_value_counts(outputs.tolist(), self.bound)
        
        if len(value_counters) >= 2:
            # Sort by occurrence frequency (descending)
            value_counters.sort(key=lambda vc: vc.get_number(), reverse=True)
            
            base_value = value_counters[0].get_value()
            self.delta = abs(value_counters[1].get_value() - base_value)
            
            if len(value_counters) > 2:
                # If the 3rd most common value is on the 'other side' of the base, 
                # use the larger of the two spreads to be conservative.
                product = (value_counters[1].get_value() - base_value) * (value_counters[2].get_value() - base_value)
                if product < 0:
                    second_diff = abs(value_counters[2].get_value() - base_value)
                    self.delta = max(self.delta, second_diff)
        
        if self.delta > 0 and limit > 0:
            # Map the physical delta to a standard deviation floor
            self.delta = self.delta * self.sscale / limit

    def calculate_sigma(self, training_set: TrainingSet, current_trend: DataTrend) -> float:
        """
        Calculates standard deviation, enforcing the discrete delta floor.

        Args:
            training_set (TrainingSet): Training data.
            current_trend (DataTrend): Trend model.

        Returns:
            float: The calculated sigma, capped at the identified delta.
        """
        sigma = super().calculate_sigma(training_set, current_trend)
        return max(sigma, self.delta)
