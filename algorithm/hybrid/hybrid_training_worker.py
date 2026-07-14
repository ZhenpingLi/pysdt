import logging
import os
import sys
from typing import List, Optional, Dict

import numpy as np

from algorithm.hybrid.hybrid_trend import HybridTrend
from algorithm.stnet.state_nn_training_worker import StateNNTrainingWorker
from algorithm.stnet.state_nn_trend import StateNNTrend
from config.sdt_constants import WARNING, ECL, MDUMP, DEFAULT
from training.preprocessing.ex_zone import ExZone
from util.time_util import DAY_IN_SECONDS, HOUR_IN_SECONDS

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config.sdt_config as sdt_config
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_trend import DataTrend
from algorithm.data_point import DataPoint
from training.training_set import TrainingSet
from training.training_worker import TrainingWorker
import algorithm.hybrid.h_pre_processing as h_processing
import training.data_buffer as data_buffer
import plugin_manager
import monitor.sdt_monitor as monitor
import algorithm.training_output_processor as top

# --- Constants ---
CONTEXT = "HybridTrainingWorker"
LINEAR = 0


def is_in_prev_session(trend: DataTrend) -> bool:
    """Checks if the trend model's window ends before the current session start."""
    pattern_times = trend.get_pattern_times()
    return pattern_times[-1] <= data_buffer.session_time if pattern_times is not None else False


def copy_list(trend: DataTrend, input_trend: DataTrend) -> None:
    """
    Copies historical statistics and normalization coefficients from a baseline 
    model to the current trend if they are temporally aligned.

    Args:
        trend (DataTrend): Target trend object.
        input_trend (DataTrend): Source baseline trend.
    """
    copied = False
    pattern_times = trend.get_pattern_times()
    if pattern_times is None:
        return
        
    num_patterns = int(len(pattern_times) / 2)
    stat_list = [None] * num_patterns
    scale_offset_list = [None] * num_patterns
    
    in_stat_list = input_trend.stat_list
    in_scale_offset_list = input_trend.scale_offset_list
    
    if in_stat_list and in_scale_offset_list:
        for index in range(num_patterns):
            ref_time = pattern_times[index * 2]
            for index_j in range(len(in_stat_list)):
                if in_stat_list[index_j]:
                    if abs(in_stat_list[index_j].time - ref_time) < 20:
                        stat_list[index] = DataPoint(ref_time, in_stat_list[index_j].data)
                        scale_offset_list[index] = DataPoint(ref_time, in_scale_offset_list[index_j].data)
                        copied = True
                        break
    if copied:
        trend.set_stats(stat_list)
        trend.set_scale_offset_list(scale_offset_list)


def _get_input_trend(inputs: List[DataTrend], trend: DataTrend) -> Optional[DataTrend]:
    """
    Identifies the most recent matching baseline trend from a list of inputs.

    Args:
        inputs (List[DataTrend]): Potential baseline models.
        trend (DataTrend): The current model being initialized.

    Returns:
        Optional[DataTrend]: The best matching baseline, or None.
    """
    best_match = None
    max_ref = 0
    for _input in inputs:
        if trend.is_match(_input):
            if _input.get_reference_time() > max_ref:
                max_ref = _input.get_reference_time()
                best_match = _input
    return best_match


def _is_training_needed(trend: DataTrend, inputs: Optional[List[DataTrend]]) -> bool:
    """
    Heuristic to determine if a state-specific model requires active fitting 
    or can reuse a baseline.

    Args:
        trend (DataTrend): Current trend.
        inputs (Optional[List[DataTrend]]): Baseline trends.

    Returns:
        bool: True if training should be performed, False otherwise.
    """
    if trend.get_sigma_t() == float('inf'):
        return False

    if inputs:
        if isinstance(trend, StateNNTrend) and is_in_prev_session(trend):
            for _input in inputs:
                if trend.is_match(_input):
                    input_times = _input.get_pattern_times()
                    pattern_times = trend.get_pattern_times()
                    if input_times is not None and pattern_times is not None:
                        if abs(input_times[0] - pattern_times[0]) < 20:
                            params = _input.params
                            if params and any(params):
                                trend.set_params(params)
                                if _input.get_stddev() > 0:
                                    trend.set_stddev(_input.get_stddev(), _input.get_stddev())
                                if trend.get_algorithm().is_monitoring_check():
                                    copy_list(trend, _input)
                                return False
    return True


def update_params(h_trend: HybridTrend, index: int) -> None:
    """
    Specialized parameter propagation for momentum dump states.
    
    If the current state is DEFAULT and follows an MDUMP state, it attempts 
    to propagate parameters.

    Args:
        h_trend (HybridTrend): The composite model.
        index (int): Index of the current trend component.
    """
    trends = h_trend.get_data_trends()
    if trends[index].get_state() == DEFAULT:
        for i in range(1, len(trends)):
            if trends[i-1].get_state() == MDUMP and trends[i].get_state() == DEFAULT:
                trends[index].set_params(trends[i].get_params())


class HybridTrainingWorker(TrainingWorker):
    """
    Coordinator for training multi-state (Finite State Machine) models.
    
    This worker manages a set of specialized sub-workers (one for each 
    operational state defined in the algorithm). It orchestrates:
    1. Dynamic state zone identification (detecting maneuvers, etc.).
    2. Data segmentation: splitting the session telemetry into state-specific sets.
    3. Parallel-style dispatching: delegating training to state sub-workers.
    4. Composite trend assembly.
    """

    def __init__(self):
        """Initializes the HybridTrainingWorker and its internal state."""
        super().__init__()
        self.midnight = float(sdt_config.get_config_value("SATMIDNIGHT") or 0.0) * HOUR_IN_SECONDS
        self.h_processing = h_processing
        self.zone_diff: float = 0.0
        self.workers: List[Optional[TrainingWorker]] = []
        self.sets: Optional[List[Optional[TrainingSet]]] = None
        self.tpc_threshold = sdt_config.get_float_property("TCWARNING") or 0.0
        self.oc_threshold = sdt_config.get_float_property("OUTLIERERRORLIMIT") or 10.0
        self.sigma_mean : Optional[Dict[str, float]] = None

    def get_algorithm_type(self) -> int:
        """Returns the algorithm type code (LINEAR)."""
        return LINEAR

    def set_config(self, node: MnemonicNode):
        """
        Configures the hybrid worker and initializes all state-specific sub-workers.

        Args:
            node (MnemonicNode): The node being trained.
        """
        super().set_config(node)
        
        state_array = self.algorithm.al_type.state
        if not state_array:
            self.workers = []
            return

        self.sigma_mean = node.sigma_mean
        self.workers = [None] * len(state_array)
        for i, state_type in enumerate(state_array):
            worker: TrainingWorker = plugin_manager.get_training_worker(state_type.algorithm)
            if worker:
                worker.set_config(node)
                worker.set_state(state_type)
                self.workers[i] = worker
                
        self.num_training = 1
        end_time = data_buffer.session_end
        start_time = end_time - 2 * self.num_pattern_in_training * self.pattern_period
        self.ref_time = start_time
        
        self.pattern_times = start_time + np.arange(2 * self.num_pattern_in_training + 1) * self.pattern_period
        
        z_diff_string = self.algorithm.get_attribute("zonediff")
        self.zone_diff = float(z_diff_string) if z_diff_string else 0.0
        self.sets = None

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Main orchestration for multi-state training.
        
        This method segments the data, determines which states require active 
        training, and dispatches to sub-workers. It also handles iterative 
        refinement for Neural Network states if TPC or outliers are too high.

        Args:
            training_set (TrainingSet): Full session telemetry.
            current_trend (DataTrend): The composite HybridTrend object.
            input_trend (Optional[DataTrend]): The composite baseline trend.
        """
        h_trend: HybridTrend = current_trend
        h_input_trend: HybridTrend = input_trend
        sub_trends = h_trend.get_data_trends()
        input_trends = h_input_trend.get_data_trends() if input_trend else None

        # 1. Update state zones based on current data features (if 'zonediff' is set)
        if self.sets is None:
            if self.zone_diff > 0:
                if self.h_processing.update_state_zone(training_set, self.zone_diff, h_trend):
                    h_trend.reset_pattern_times()
        else:
            if self.zone_diff > 0:
                for index in range(len(sub_trends)):
                    sub_trends[index].set_pattern_times(input_trends[index].get_pattern_times() if input_trends else None)
                    
        # 2. Segment full telemetry into state-specific TrainingSets
        if self.sets is None or self.dqf_changed:
            if self.dqf_changed:
                h_trend.get_algorithm().set_check_dqf(True)
            self.sets = self.h_processing.get_training_sets(training_set, h_trend)
        else:
            # Refresh statistics for existing sets
            for i, trend in enumerate(sub_trends):
                if self.sets[i]:
                    trend.set_stats(self.sets[i].stat_list)

        # 3. Train each state component
        for st_index, trend in enumerate(sub_trends):
            worker = self._get_worker(trend.get_state())
            if worker and self.sets[st_index]:
                input_sub_trend = _get_input_trend(input_trends, trend) if input_trends else None
                prev_sigma = input_sub_trend.get_stddev() if input_sub_trend else 0.0
                
                if _is_training_needed(trend, input_trends):
                    if trend.get_stat_list() is None:
                        trend.set_stats(self.sets[st_index].stat_list)
                        if self.sets[st_index].coef_list:
                            trend.set_scale_offset_list(self.sets[st_index].coef_list)
                    
                    avg_sigma = self.sigma_mean.get(worker.state, 0.0) if self.sigma_mean else 0.0
                    
                    # Core sub-training dispatch
                    worker.do_data_training(self.sets[st_index], trend, input_sub_trend)
                    sigma = worker.calculate_sigma(self.sets[st_index], trend)

                    # High-fidelity refinement for Neural Network states
                    if avg_sigma > 0.0:
                        if isinstance(worker, StateNNTrainingWorker):
                            if self._need_refresh(sigma, self.sets[st_index], trend, prev_sigma):
                                worker.set_refresh(True)
                                worker.do_data_training(self.sets[st_index], trend, input_trend)
                                sigma = worker.calculate_sigma(self.sets[st_index], trend)

                    logging.info(f"{CONTEXT}: Sigma for {trend.get_state()} state: {sigma:.4f}")
                    trend.set_stddev(sigma, avg_sigma)
                else:
                    # Reuse baseline parameters but update statistics
                    if trend.get_stat_list() is None:
                        trend.set_stats(self.sets[st_index].stat_list)
                        if self.sets[st_index].coef_list:
                            trend.set_scale_offset_list(self.sets[st_index].coef_list)

                trend.set_trended(True)

    def _get_worker(self, state_name: str) -> Optional[TrainingWorker]:
        """Returns the worker instance assigned to the specific state."""
        for worker in self.workers:
            if worker and worker.state == state_name:
                return worker
        return None

    def do_monitoring(self, input_trend: DataTrend, training_set: TrainingSet, ptimes: np.ndarray):
        """
        Executes outlier detection and DQF initialization for hybrid models.

        Args:
            input_trend (DataTrend): Baseline HybridTrend.
            training_set (TrainingSet): Data to monitor.
            ptimes (np.ndarray): Pattern timestamps.
        """
        h_trend: HybridTrend = input_trend
        
        if self.zone_diff > 0:
            if self.h_processing.update_state_zone(training_set, self.zone_diff, h_trend):
                h_trend.reset_pattern_times()
            
        extended_zone = self._get_extended_zone(h_trend.state_zones)
        self.sets = self.h_processing.get_training_sets(training_set, h_trend)
        
        for index, (_set, trend) in enumerate(zip(self.sets, h_trend.trends, strict=True)):
            if _set and trend:
                ecl_type = next((st for st in self.algorithm.al_type.state if st.name == ECL), None)
                dim_pointer = 0
                if ecl_type and ecl_type.dim_pointer:
                    dim_pointer = ecl_type.dim_pointer
                
                # Specialized logic for default/eclipse state DQF initialization
                if _set.state == DEFAULT and (h_trend.get_state_trend(ECL) or dim_pointer == 0):
                    if trend.get_params() is None and h_trend.is_disjoint:
                        update_params(h_trend, index)
                    if trend.get_params() is not None:
                        current_zone = extended_zone if _set.state == DEFAULT else self.ex_zone
                        monitor.init_dqf(trend, _set, trend.get_pattern_times(), current_zone)

    def _get_extended_zone(self, zones: List[ExZone]) -> Optional[ExZone]:
        """
        Predicts exclusion zones for future timestamps by shifting historical zones.
        Used for real-time monitoring.

        Args:
            zones (List[ExZone]): Historical zones.

        Returns:
            Optional[ExZone]: The shifted and merged extended zone.
        """
        c_zone = None
        for i in range(1, len(zones)):
            if zones[i]:
                zone_in_period = self.h_processing.get_ex_zones(zones[i], data_buffer.session_start, data_buffer.session_time, None)
                if zone_in_period:
                    if c_zone is None:
                        c_zone = zone_in_period
                    else:
                        c_zone.merge(zone_in_period, True)
        if c_zone:
            # Shift historical zones forward by one day
            zone_periods = c_zone.get_zones() + DAY_IN_SECONDS
            c_zone.ex_zones = zone_periods
            if self.ex_zone:
                c_zone.merge(self.ex_zone, True)
            
        return c_zone

    def _need_refresh(self, sigma: float, training_set: TrainingSet, trend: DataTrend, prev_sigma: float) -> bool:
        """
        Determines if a Neural Network model requires high-precision refinement 
        due to significant temporal change or excessive outliers.
        """
        if not isinstance(trend, StateNNTrend) or trend.get_num_pattern_in_training() == 2:
            return False
            
        if prev_sigma > 0:
            tpc = sigma / prev_sigma
            if tpc > self.tpc_threshold:
                return True
                
        # Also check for large outlier clusters
        outlier_list = monitor.check_outlier(trend, training_set, WARNING, self.ex_zone)
        if outlier_list:
            event_data_list = top.create_mnemonic_event_data_list_from_outlier_list(outlier_list)
            outlier_value = top.get_oc_value(event_data_list, None, None, training_set.state)
            return outlier_value > self.oc_threshold
            
        return False

    def _perform_training(self, trend: DataTrend) -> bool:
        """Forces training for all sub-components of a hybrid model."""
        return True
