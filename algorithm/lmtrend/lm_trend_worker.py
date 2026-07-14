import logging
import os
import sys
from typing import Optional

import numpy as np

from algorithm.mnemonic_node import MnemonicNode
from training.preprocessing import data_stat
from util import sdt_util

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.training_worker import TrainingWorker
from algorithm.data_trend import DataTrend
from training.training_set import TrainingSet
from training import data_buffer
from config.sdt_constants import LINEAR, MEAN_INDEX
from algorithm.lmtrend import lm_preprocessor as lm_prep

CONTEXT = "LMTRAINING"

class LMTrendWorker(TrainingWorker):
    """
    A specialized training worker for the limit-based (LM) trending algorithm.
    Calculates a simple statistical mean of the data.
    """

    def __init__(self):
        super().__init__()

    def get_algorithm_type(self) -> int:
        return LINEAR

    def set_config(self, node: MnemonicNode):
        super().set_config(node)

        self.ref_time = data_buffer.session_start
        self.num_training = 1
        self.num_pattern_in_training = 1
        
        start = data_buffer.session_start
        end = data_buffer.session_end
        self.pattern_times = np.array([start, (start + end) / 2.0, end])
        self.pattern_period = (end - start) / 2.0

    def do_training(self, training_set: TrainingSet, node: MnemonicNode):
        """
        Performs the high-level training flow for the limit-based model.
        """
        self.set_config(node)
        
        input_trend = node.input_trend
        prev_sigma = input_trend.get_stddev() if input_trend else 0.0
        pattern_offset_index = 0
        
        # 1. Prepare Training Set
        delta_set = lm_prep.get_lm_training_set(training_set, self.algorithm)
        training_set = self.processing.preprocess_training_set(delta_set, node, self.pattern_times, self.pattern_period, 0)

        
        logging.info(f"{CONTEXT}: The training sample size is: {len(delta_set.outputs)}")
        
        if input_trend:
            # Initialize DQF using monitor
            self._do_monitoring(node.input_trend, training_set, self.pattern_times)

        # 2. Create Trend Object
        trends = [self._create_data_trend(node, 0, 0)]
        current_trend = trends[0]
        
        p_times = current_trend.pattern_times
        sub_set = sdt_util.get_subset_by_time(training_set, p_times[0], p_times[-1])
        current_trend.set_stats(sub_set.stat_list)
        
        # 3. Perform Fitting
        sigma_t = current_trend.get_sigma_t()
        # Note: assuming algorithm_type check
        if sigma_t > 0:
            self.do_data_training(sub_set, current_trend, input_trend)
            sigma = self.calculate_sigma(sub_set, current_trend)
            current_trend.set_stddev(sigma, prev_sigma)
            
            # 4. Outlier Monitoring and Retraining
            if self._do_monitoring(current_trend, sub_set, self.pattern_times):
                self.dqf_changed = True
                self.do_data_training(sub_set, current_trend, input_trend)
                sigma = self.calculate_sigma(sub_set, current_trend)
                if self.min_std > 0 and sigma < self.min_std:
                    sigma = self.min_std
                current_trend.set_stddev(sigma, prev_sigma)
            logging.info(f"{CONTEXT}: The Sigma Value for {node.name} is {sigma:.4f}")
            current_trend.set_stddev(sigma, prev_sigma)
            
            logging.info(f"{CONTEXT}: Final Sigma Value for {node.name} is {sigma:.4f}")

        node.data_trend=trends

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend] = None):
        """
        The core data training logic: calculates the mean.
        """
        stat_dp = data_stat.cal_stat(training_set, data_buffer.session_start)
        
        params = [float(stat_dp.data[MEAN_INDEX])]
        current_trend.set_params(params)
