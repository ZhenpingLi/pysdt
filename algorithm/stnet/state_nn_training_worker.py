import logging
import os
import sys
from typing import List, Optional

import numpy as np

from config.sdt_constants import SIGMA_INDEX, DAY_IN_SECONDS
from sdtdb import sdt_db
from sdtdb.sdt_db import StateType
from training.preprocessing.lttb_filter import LTTBFilter
from training.preprocessing.pre_processing import get_nn_training_set
from util import time_util
from .state_nn_trend import StateNNTrend
from ..algorithm_def import AlgorithmDef

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_trend import DataTrend
from algorithm.data_point import DataPoint
from training.training_set import TrainingSet
from algorithm.fbnn.nn_training_worker import NNTrainingWorker
from algorithm.stnet.state_pre_processing import StatePreProcessing
from algorithm.stnet import state_nn_trend_param_io as param_io


# --- Constants ---
CONTEXT = "StateNNTrainingWorker"


def _is_data_constant(stats: Optional[List[DataPoint]]) -> bool:
    """
    Checks if the telemetry data is effectively constant across all pattern cycles.

    Args:
        stats (Optional[List[DataPoint]]): Descriptive statistics per cycle.

    Returns:
        bool: True if data is constant (sigma=0) or missing, False otherwise.
    """
    if not stats:
        return True
    for stat in stats:
        if stat and stat.data[SIGMA_INDEX] > 0:
            return False
    return True


def _is_null(params: Optional[np.ndarray]) -> bool:
    """
    Heuristic to check if a parameter array is uninitialized or effectively zero.

    Args:
        params (Optional[np.ndarray]): Model weights array.

    Returns:
        bool: True if params are None or predominantly zero, False otherwise.
    """
    if params is None:
        return True
    # Check if the first half of params (weights) are all zero
    return not np.any(params[:len(params)//2])


class StateNNTrainingWorker(NNTrainingWorker):
    """
    Concrete TrainingWorker implementation for State-based Neural Networks (STNET).
    
    This worker fits Neural Network models specialized for specific operational 
    states. It coordinates state-specific preprocessing, retrieves baseline 
    parameters from local storage, and executes high-fidelity fitting with 
    L-BFGS optimization.
    """

    def __init__(self):
        """Initializes the StateNNTrainingWorker."""
        super().__init__()
        self.pre_processing = StatePreProcessing()
        self.dim_offset = 0
        self.refresh = False
        self.nn_struct = None
        self.pattern_period = DAY_IN_SECONDS

    def set_config(self, node: MnemonicNode):
        """
        Configures the worker with algorithm metadata and initialization context.

        Args:
            node (MnemonicNode): The node being trained.
        """
        self.algorithm = AlgorithmDef(sdt_db.get_algorithm(node.name))
        self.num_pattern_in_training = 1
        self.refresh = False
        
        socoef_query = "socoef"
        socoef_string = self.algorithm.get_attribute(socoef_query)
        if socoef_string:
            tokens = socoef_string.split('|')
            socoef = np.array([float(tokens[0]), float(tokens[1])], dtype=np.float32)
            self.processing.set_so_coef(socoef)

        self._init_training.set_config(self.algorithm, node.name)

    def set_refresh(self, r: bool):
        """
        Sets the refresh flag to force a full re-initialization of weights.

        Args:
            r (bool): If True, baseline parameters are ignored.
        """
        self.refresh = r

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Executes the specialized STNET training pipeline.
        
        This method manages:
        1. State-specific normalization (StatePreProcessing).
        2. Parameter retrieval from local weight archives.
        3. Data downsampling using LTTB.
        4. Incremental retraining or global initialization.
        5. Exporting updated weights to local storage.

        Args:
            training_set (TrainingSet): Data points for the specific state.
            current_trend (DataTrend): The model object to populate.
            input_trend (Optional[DataTrend]): Baseline model.
        """
        nn_trend: StateNNTrend = current_trend
        in_trend = input_trend
        self.nn_struct = nn_trend.nn_struct
        self._init_training.set_nn_struct(self.nn_struct)
        
        if _is_data_constant(training_set.stat_list):
            return

        if current_trend.get_stat_list() is None:
            current_trend.set_stats(training_set.stat_list)
            current_trend.set_scale_offset_list(training_set.coef_list)

        init_params = input_trend.get_params() if input_trend else None
        
        # Retrieval from local parameter archive if baseline is missing
        if not self.refresh and (_is_null(init_params) or (in_trend and self.num_pattern_in_training == 1)):
            init_params = param_io.input_data_trend(current_trend)

        self.ref_time = current_trend.get_reference_time()
        
        if training_set.coef_list is None:
            norm_type = self.algorithm.get_attribute("normtype")
            self.pre_processing.transform(training_set, norm_type)
            current_trend.set_scale_offset_list(training_set.coef_list)
            
        nn_set = get_nn_training_set(training_set, current_trend)
        if not nn_set:
            return

        logging.info(f"[{self.mnemonic_id}] {CONTEXT}: Training with {len(nn_set.outputs)} samples.")
        
        sub_set = nn_set
        is_new_training = False
        
        if init_params is None or len(init_params) != nn_trend.get_param_dim():
            # Perform initial global search
            if len(nn_set.outputs) > 180:
                _filter = LTTBFilter(n_out=180)
                sub_set = _filter.filter(nn_set)
            self.init_training(sub_set, nn_trend, training_set)
            is_new_training = True
        else:
            # Perform incremental refinement
            if len(nn_set.outputs) > 2400:
                _filter = LTTBFilter(n_out=2400)
                sub_set = _filter.filter(nn_set)
            min_weight = self.retrain(sub_set, init_params)
            current_trend.set_params(min_weight.tolist() if isinstance(min_weight, np.ndarray) else min_weight)
            
        current_trend.set_trended(True)

        if is_new_training:
            param_io.export_data_trend(current_trend)

    def set_state(self, s_type: StateType):
        """
        Registers the target operational state and configures the network topology.

        Args:
            s_type (StateType): The state configuration including dimensionality info.
        """
        super().set_state(s_type)
        _offset = s_type.dim_pointer
        if _offset is not None:
            self.dim_offset = int(_offset)
            
        dimensions = self.algorithm.get_dimension()
        # Define structure based on state-specific dimensionality
        self.struct = [1, dimensions[self.dim_offset], dimensions[self.dim_offset+1], 1]
        self._init_training.set_nn_struct(self.struct)
