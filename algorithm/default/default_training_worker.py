import logging
import os
import sys
from typing import List, Optional

import numpy as np

from config import sdt_config
from config.sdt_constants import MEAN_INDEX, SHORTTERM
from monitor import sdt_monitor
from training import data_buffer
from training.preprocessing import data_stat
from util import sdt_util

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.training_worker import TrainingWorker
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_point import DataPoint
from algorithm.data_trend import DataTrend
from training.training_set import TrainingSet

# Constants
CONTEXT = "DefaultTrainingWorker"
LINEAR = 0


class DefaultTrainingWorker(TrainingWorker):
    """
    Concrete TrainingWorker implementation for the 'default' training algorithm.
    
    This worker specializes in modeling telemetry that is generally constant. 
    It performs training by:
    1. Calculating simple descriptive statistics (mean, sigma) for the session.
    2. Using the calculated arithmetic mean as the primary model parameter.
    3. Performing iterative outlier detection and re-calculation of the mean.
    """

    def __init__(self):
        """Initializes the DefaultTrainingWorker."""
        super().__init__()
        self.stat_point: Optional[DataPoint] = None
        self.min_sigma: float = 0.0

    def get_algorithm_type(self) -> int:
        """Returns the algorithm type code (LINEAR)."""
        return LINEAR

    def set_config(self, node: MnemonicNode):
        """
        Customizes the temporal configuration for the default algorithm.
        
        The default algorithm uses a fixed two-cycle pattern (48h total) 
        terminating at the current session end.

        Args:
            node (MnemonicNode): The mnemonic node to be trained.
        """
        super().set_config(node)
        
        self.ref_time = data_buffer.session_start
        self.num_training = 1
        self.num_pattern_in_training = 1
        
        session_period = sdt_config.session_period
        self.pattern_period = session_period / 2.0
        
        # Define the three timestamps for two 24h pattern cycles
        self.pattern_times = np.array([
            data_buffer.session_end - 2 * self.num_pattern_in_training * self.pattern_period,
            (data_buffer.session_end - self.num_pattern_in_training * self.pattern_period),
            data_buffer.session_end
        ])
        
        self.min_sigma = float(self.algorithm.get_attribute("minstd") or 0.0)


    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Calculates the arithmetic mean of the dataset and sets it as the 
        model parameter.

        Args:
            training_set (TrainingSet): Data points to train on.
            current_trend (DataTrend): The model object to update.
            input_trend (Optional[DataTrend]): The baseline model (unused for fitting).
        """
        self.stat_point = data_stat.cal_stat(training_set, self.ref_time, None)
        
        params = np.zeros(1)
        if self.stat_point:
            params[0] = self.stat_point.data[MEAN_INDEX]
            
        logging.info(f"[{self.mnemonic_id}] {CONTEXT}: Calculated session mean: {params[0]:.4f}")
        
        current_trend.set_params(params)
        current_trend.set_trended(True)

    def do_training(self, training_set: TrainingSet, node: MnemonicNode):
        """
        Orchestrates the iterative training and outlier detection loop for 
        the default algorithm.

        Args:
            training_set (TrainingSet): Raw input telemetry.
            node (MnemonicNode): The mnemonic node in the tree.
        """
        self.set_config(node)
        input_trend = node.input_trend
        prev_sigma = 0.0
        
        # 1. Initial Preprocessing (stat calculation)
        training_set = self.processing.preprocess_training_set(training_set, node, self.pattern_times, self.pattern_period, 0)
        
        if input_trend and data_buffer.session_type == SHORTTERM:
            prev_sigma = input_trend.get_stddev()
            if self.min_sigma > 0 and prev_sigma < self.min_sigma:
                input_trend.set_stddev(self.min_sigma, prev_sigma)
            # Initialize DQF mask based on previous session's results
            sdt_monitor.init_dqf(input_trend, training_set, self.pattern_times, None)

        # 2. Setup the target trend model
        trends = [self._create_data_trend(node, 0, 0)]
        p_times = trends[0].get_pattern_times()
        sub_set = sdt_util.get_subset_by_time(training_set, p_times[0], p_times[-1])
        
        if not sub_set:
            trends[0].set_trended(True)
            node.data_trend = trends
            return

        trends[0].set_stats(sub_set.stat_list)
        
        sigma_t = trends[0].get_sigma_t()
        if sigma_t > 0 and sigma_t != float('inf'):
            # Round 1: Training on initial DQF
            self.do_data_training(sub_set, trends[0], input_trend)
            sigma = self.calculate_sigma(sub_set, trends[0])
            
            if self.min_std > 0 and sigma < self.min_std:
                sigma = self.min_std
            
            trends[0].set_stddev(sigma, prev_sigma)
            
            # Round 2: Detect Outliers and Retrain if necessary
            if sdt_monitor.monitor(node, trends[0], training_set, self.ex_zone):
                logging.info(f"{CONTEXT}: Outliers identified for {self.mnemonic_id}. Executing retraining round.")
                self.do_data_training(sub_set, trends[0], input_trend)
                sigma = self.calculate_sigma(sub_set, trends[0])
                if self.min_std > 0 and sigma < self.min_std:
                    sigma = self.min_std
                trends[0].set_stddev(sigma, prev_sigma)
            
            logging.info(f"{CONTEXT}: Final standard deviation for {self.mnemonic_id}: {sigma:.4f}")
        else:
            # Handle static data cases
            trends[0].set_stddev(sigma_t, prev_sigma)
            if trends[0].get_stat_list() is None:
                trends[0].set_stats(sub_set.stat_list)
            
            if sigma_t != float('inf'):
                if sdt_monitor.monitor(node, trends[0], sub_set, self.ex_zone):
                    logging.info(f"{CONTEXT}: Outliers identified in otherwise static data for {self.mnemonic_id}")
                else:
                    logging.info(f"{CONTEXT}: Data for {self.mnemonic_id} is static.")

        trends[0].set_trended(True)
        node.data_trend = trends
