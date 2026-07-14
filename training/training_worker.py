import gc
import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np

from config import sdt_config
from config.sdt_constants import TREND, DEFAULT, LONGTERM, HYBRID
from monitor import sdt_monitor as sdt_monitor
from sdtdb import sdt_db
from sdtdb.sdt_db import StateType
from training.preprocessing.state_zones import StateZones, CURRENT
from util import sdt_util
from util.typed_list import TypedList

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_trend import DataTrend
from algorithm.algorithm_def import AlgorithmDef
from training.training_set import TrainingSet
from training.preprocessing.long_term_processing import LongTermProcessing
from training.preprocessing.short_term_processing import ShortTermProcessing
from training.preprocessing.ex_zone import ExZone
from training import data_buffer
from util.time_util import get_time_tag_from_seconds
from algorithm.hybrid.hybrid_trend import HybridTrend

# Constants
CONTEXT = "TrainingWorker"
MINSTD = "minstd"
SOCOEF = "socoef"
STATIC = 2


class TrainingWorker(ABC):
    """
    Abstract base class for data training algorithms.
    
    This class defines the skeleton of the training process:
    1. Configuration setup and temporal context initialization.
    2. Preprocessing of telemetry data (e.g., normalization, gap checking).
    3. Execution of the specific training algorithm (implemented by subclasses).
    4. Iterative outlier detection and model refinement.
    5. Statistical performance evaluation (e.g., sigma/TPC calculation).
    """

    def __init__(self):
        """
        Initializes the TrainingWorker and selects the appropriate 
        preprocessing strategy based on session type.
        """
        if data_buffer.session_type == LONGTERM:
            self.processing = LongTermProcessing()
        else:
            self.processing = ShortTermProcessing()
        
        self.num_pattern_in_training = sdt_config.num_pattern_in_training
        self.ex_zone: Optional[ExZone] = None
        self.state: str = DEFAULT
        self.mnemonic_id: str = ""
        self.ref_time: float = 0.0
        self.pattern_period: float = 0.0
        self.num_training: int = 1
        self.algorithm: Optional[AlgorithmDef] = None
        self.pattern_times: Optional[np.ndarray] = None
        self.algorithm_type: int = 0
        self.min_std: float = -0.1
        self.tpc_trending: bool = False
        self.dqf_changed: bool = False

    def set_config(self, node: MnemonicNode):
        """
        Configures the worker with parameters specific to the given mnemonic node.
        
        It retrieves algorithm definitions, model times, and exclusion zones 
        from the database and session context.

        Args:
            node (MnemonicNode): The tree node to be trained.
        """
        self.mnemonic_id = node.name
        self.algorithm = AlgorithmDef(sdt_db.get_algorithm(self.mnemonic_id))
        model_time = data_buffer.get_session_model_time(self.algorithm)
        
        self.pattern_times = model_time.get_pattern_times()
        self.pattern_period = model_time.get_model_period()
        self.ref_time = model_time.get_reference_time()
        
        if self.algorithm.get_np() > 0:
            self.num_pattern_in_training = self.algorithm.get_np()

        self.min_std = float(self.algorithm.get_attribute(MINSTD) or 0.0)

        socoef_string = self.algorithm.get_attribute(SOCOEF)
        if socoef_string:
            tokens = socoef_string.split('|')
            socoef = np.array([float(tokens[0]), float(tokens[1])], dtype=np.float32)
            self.processing.set_so_coef(socoef)
            
        self.ex_zone = ExZone.get_ez_zone(self.algorithm, self.ref_time, data_buffer.session_end)
        self.tpc_trending = self.algorithm.get_attribute("tpctrend")
        self.dqf_changed = False

    def do_training(self, training_set: TrainingSet, node: MnemonicNode):
        """
        Coordinates the training loop for a mnemonic.
        
        This method performs the following steps:
        1. Preprocesses the training set.
        2. Creates trend model objects for the required number of windows.
        3. Executes the core algorithm training.
        4. Triggers outlier monitoring and re-trains if the dataset changes.
        5. Calculates final model statistics.

        Args:
            training_set (TrainingSet): The raw and processed telemetry data.
            node (MnemonicNode): The node representing the mnemonic in the tree.
        """
        self.set_config(node)
        logging.info(f"{CONTEXT}: Data Training for {self.mnemonic_id} ref time: {get_time_tag_from_seconds(self.ref_time)}")
        
        prev_sigma = 0.0
        pattern_offset_index = 0
        input_trend = node.input_trend
        
        if input_trend and np.array_equal(node.name, input_trend.mnemonic_id):
            prev_sigma = input_trend.get_stddev()
            pattern_offset_index = self.get_pattern_offset_index(input_trend.get_reference_time())

        # Baseline for Temporal Change (TPC) comparison
        if self.tpc_trending and self.algorithm.get_name() != HYBRID:
            prev_sigma = node.sigma_mean.get('default', 0.0)
            
        self.num_training = int(((len(self.pattern_times) - pattern_offset_index - 1) / self.num_pattern_in_training) + 0.5) - 1
        if self.num_training <= 0:
            self.num_training = 1
            
        trends = TypedList(self.num_training, DataTrend)
        training_set = self.processing.preprocess_training_set(training_set, node, self.pattern_times, self.pattern_period, pattern_offset_index)
        
        if data_buffer.is_manual_training:
            node.input_trend = None
            
        if training_set.outputs is not None:
            for i in range(self.num_training):
                trends[i] = self._create_data_trend(node, pattern_offset_index, i)
                p_times = trends[i].get_pattern_times()
                if p_times is None or len(p_times) == 0:
                    continue

                sub_set = sdt_util.get_subset_by_time(training_set, p_times[0], p_times[-1])
                if not sub_set:
                    continue
                
                trends[i].set_stats(sub_set.stat_list)
                trends[i].set_scale_offset_list(sub_set.coef_list)
                
                if self._perform_training(trends[i]):
                    if self.algorithm.is_monitoring_check() and node.input_trend:
                        self._do_monitoring(node.input_trend, sub_set, p_times)
                    
                    # 1. Initial Training Round
                    self.do_data_training(sub_set, trends[i], node.input_trend)
                    sigma = self.calculate_sigma(sub_set, trends[i])
                    
                    if self.min_std > 0 and sigma < self.min_std:
                        sigma = self.min_std
                        
                    trends[i].set_stddev(sigma, prev_sigma)
                    
                    # 2. Outlier Monitoring and Iterative Refinement
                    monitor_set = training_set if self.num_training == 1 else sub_set
                    if sdt_monitor.monitor(node, trends[i], monitor_set, self.ex_zone, self.pattern_times):
                        self.dqf_changed = True
                        # Re-train using the updated DQF mask
                        self.do_data_training(sub_set, trends[i], trends[i])
                        sigma = self.calculate_sigma(sub_set, trends[i])
                        if self.min_std > 0 and sigma < self.min_std:
                            sigma = self.min_std
                        trends[i].set_stddev(sigma, prev_sigma)
                        
                    logging.info(f"{CONTEXT}: Resulting Sigma for {self.mnemonic_id} is {sigma:.4f}")
                    node.input_trend = trends[i]
                    prev_sigma = sigma
                else:
                    logging.info(f"{CONTEXT}: Sigma is zero or invalid, training skipped for {self.mnemonic_id}")
                    trends[i].set_stats(sub_set.stat_list)
                
                trends[i].set_trended(True)
        
        node.data_trend = trends.data
        gc.collect()

    def _create_data_trend(self, node: MnemonicNode, pattern_offset_index: int, i: int) -> DataTrend:
        """Creates a new DataTrend object with the appropriate pattern time context."""
        from plugin_manager import get_data_trend
        trend = get_data_trend(self.algorithm.get_name(), node.name)
        
        p_times_size = 2 * self.num_pattern_in_training + 1
        exist_size = len(self.pattern_times) - i * self.num_pattern_in_training - pattern_offset_index
        
        if p_times_size > exist_size >= 2:
            p_times_size = exist_size
            
        offset = pattern_offset_index + i * self.num_pattern_in_training
        if offset < len(self.pattern_times):
            if offset + p_times_size <= len(self.pattern_times):
                p_times = self.pattern_times[offset : offset + p_times_size]
            else:
                p_times = self.pattern_times[offset:].tolist()
                for j in range(p_times_size - len(p_times)):
                    p_times.append(self.pattern_times[-1] + (j + 1) * self.pattern_period)
                p_times = np.array(p_times)
                
        trend.set_data_model_time(p_times, self.pattern_period)
        if trend.get_alg_name() == HYBRID:
            state_zone = StateZones(trend.algorithm, trend.mnemonic_id)
            h_trend : HybridTrend = trend
            h_trend.set_state_zones(state_zone.init_state_zones(CURRENT))

        return trend

    def get_pattern_offset_index(self, input_ref_time: float) -> int:
        """Determines the starting pattern cycle relative to the input model."""
        return 0

    def set_state(self, state: StateType):
        """Sets the active operational state for the worker."""
        self.state = state.name

    @abstractmethod
    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Executes the algorithm-specific model fitting. 
        Must be implemented by concrete subclasses.

        Args:
            training_set (TrainingSet): The data points to train on.
            current_trend (DataTrend): The model object being updated.
            input_trend (Optional[DataTrend]): The baseline model for incremental updates.
        """
        pass

    @abstractmethod
    def get_algorithm_type(self) -> int:
        """Returns the type of algorithm (e.g., LINEAR, NONLINEAR)."""
        pass

    def _do_monitoring(self, input_trend: DataTrend, training_set: TrainingSet, pattern_times: np.ndarray):
        """Initializes Data Quality Flags (DQF) based on the previous session's model."""
        sdt_monitor.init_dqf(input_trend, training_set, pattern_times, self.ex_zone)

    def calculate_sigma(self, training_set: TrainingSet, current_trend: DataTrend) -> float:
        """
        Calculates the standard deviation of the model fit using vectorized operations.
        
        It calculates residuals only for valid data points (DQF=1) and returns 
         the Root Mean Square Error (RMSE).

        Args:
            training_set (TrainingSet): The data set containing raw values and DQF mask.
            current_trend (DataTrend): The model to evaluate.

        Returns:
            float: The calculated standard deviation (RMSE).
        """
        inputs = training_set.inputs
        results = training_set.raw
        dqf = training_set.dqf

        valid_mask = (dqf == 1)
        if not np.any(valid_mask):
            logging.error(f"{CONTEXT}: Sigma calculation failed; no valid data points.")
            trend_std = current_trend.get_stddev()
            return float(trend_std) if trend_std > 0 else 0.0

        valid_inputs = inputs[valid_mask]
        valid_results = results[valid_mask]

        # Vectorized prediction across all valid points
        predicted_values = current_trend.get_all_trend_values(valid_inputs, TREND)
        if predicted_values is None:
            return 0.0

        residuals = valid_results - predicted_values[:, 0]
        mse = np.mean(np.square(residuals))
        sigma = np.sqrt(mse)

        return float(sigma) if not np.isnan(sigma) else 0.0

    def _perform_training(self, trend: DataTrend) -> bool:
        """Checks if training should be executed based on statistical validity."""
        sigma_t = trend.get_sigma_t()
        return (sigma_t > 0 and sigma_t != float('inf')) or self.algorithm_type == STATIC
