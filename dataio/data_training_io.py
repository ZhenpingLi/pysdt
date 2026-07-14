import logging
import os
import sys
import traceback
from typing import List, Optional

import orbit.orbit_model_manager as omm
from algorithm.algorithm_data import AlgorithmData
from algorithm.algorithm_def import AlgorithmDef
from algorithm.data_point import DataPoint
from algorithm.hybrid_state_data import HybridStateData
from algorithm.training_output import TrainingOutputData
from config.sdt_constants import DAY_IN_SECONDS, MNVR, HOUR_IN_SECONDS, CURRENT, PREV
from dataio.sdt_data_io import SDTDataTrainingIO
from sdtdb import sdt_db
from training import data_buffer

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.data_trend import DataTrend
from algorithm.hybrid.hybrid_trend import HybridTrend
from algorithm.stnet.state_nn_trend import StateNNTrend
import training.data_buffer as db
import plugin_manager

# --- Constants ---
CONTEXT = "DataTrainingIO"
DEFAULT = "default"
HYBRID = "hybrid"

sdt_data_io : SDTDataTrainingIO = None


def _get_query_range(prev_trend: DataTrend, in_time: List[float], state: str) -> Optional[List[float]]:
    """
    Determines the appropriate time range for querying historical data for a specific trend type.
    
    Args:
        prev_trend (DataTrend): The trend object from the previous session.
        in_time (List[float]): User-specified input time window.
        state (str): The operational state name.

    Returns:
        Optional[List[float]]: A list [reference_time, window_size], or None.
    """
    _in_time = None
    if prev_trend:
        if isinstance(prev_trend, StateNNTrend):
            _in_time = [0.0, 0.0]
            _in_time[0] = prev_trend.get_reference_time()
            _in_time[1] = HOUR_IN_SECONDS
        else:
            _in_time = [0.0, 0.0]
            _np = prev_trend.get_num_pattern_in_training()
            _in_time[0] = db.session_time - 2*_np*DAY_IN_SECONDS
            _in_time[1] = DAY_IN_SECONDS
    else:
        if state == MNVR:
            _in_time = [0.0, 0.0]
            _in_time[0] = db.session_time - 4 * DAY_IN_SECONDS
            _in_time[1] = DAY_IN_SECONDS

    if in_time is not None:
        _in_time = in_time

    return _in_time


def _get_prev_trend(prev_trend_container: HybridTrend, current_trend: DataTrend) -> Optional[DataTrend]:
    """
    Finds the matching trend from the previous session for a given current trend component.
    
    Args:
        prev_trend_container (HybridTrend): The collection of trends from the previous session.
        current_trend (DataTrend): The trend being initialized for the current session.

    Returns:
        Optional[DataTrend]: The matching previous trend, or None if no match found.
    """
    data_trends = prev_trend_container.get_data_trends()
    _prev_trend = None
    for trend in data_trends:
        if trend.is_disjoint:
            ref_time = current_trend.get_reference_time()
            prev_ref_time = trend.get_reference_time()
            if abs(ref_time-prev_ref_time) < 20:
                _prev_trend = trend
                break
        else:
            if trend.is_match(current_trend):
                _prev_trend = trend
                break
    return _prev_trend

def create_hybrid_trend(mnemonic_id: str, session_index: int) -> HybridTrend:
    """
    Factory function to initialize a HybridTrend object with appropriate state zones and model times.
    
    Args:
        mnemonic_id (str): The identifier for the mnemonic.
        session_index (int): Indicates either CURRENT or PREV session context.

    Returns:
        HybridTrend: The initialized hybrid trend object.
    """
    h_trend: HybridTrend = plugin_manager.get_data_trend(HYBRID, mnemonic_id)
    if session_index == CURRENT:
        model_time = data_buffer.get_default_model_time(h_trend.algorithm)
    else:
        model_time = omm.create_default_model_time_for_prev_session(h_trend.algorithm)
    ez_zone = data_buffer.get_state_zones(h_trend.algorithm, mnemonic_id, session_index)
    h_trend.set_data_model_time(model_time.pattern_times, model_time.model_period)
    h_trend.set_state_zones(ez_zone)
    return h_trend


class DataTrainingIO:
    """
    High-level I/O manager for SDT training results.
    
    This class orchestrates the loading of baseline trends for new training 
    sessions and the archival of trained models into persistent storage 
    using pluggable I/O components.
    """

    def __init__(self):
        """
        Initializes the DataTrainingIO by loading the default storage plugin.
        """
        self.input_trend_set = False
        self.sdt_data_io : SDTDataTrainingIO = plugin_manager.get_sdt_data_training_io("default")
        logging.info(f"{CONTEXT}: Initialized the data training IO as a plugin component.")

    def __enter__(self):
        """Supports the context manager pattern."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures the underlying storage connection is closed when exiting context."""
        self.close()

    def save(self, training_output_list: List[TrainingOutputData]):
        """
        Persists a list of training results to the archive.

        Args:
            training_output_list (List[TrainingOutputData]): The collection of 
                results to be saved.
        """
        for training_output in training_output_list:
            if training_output.training_error is None:
                try:
                    self.sdt_data_io.write_data_trend(training_output.algorithm_data_list)
                except Exception as e:
                    error_details = traceback.format_exc()
                    logging.error(f"{CONTEXT}: Error writing data output for {training_output.mnemonic_id}: {error_details}")

    def get_input_data_trend(self, mnemonic_id: str, in_time: List[float]) -> Optional[AlgorithmData]:
        """
        Retrieves the baseline model (AlgorithmData) for a mnemonic to initialize training.
        Handles both simple and hybrid (multi-state) trend models.

        Args:
            mnemonic_id (str): ID of the mnemonic.
            in_time (List[float]): Preferred time range for the baseline.

        Returns:
            Optional[AlgorithmData]: The baseline parameters, or None if not found.
        """
        algorithm = AlgorithmDef(sdt_db.get_algorithm(mnemonic_id))
        if algorithm.get_name() == HYBRID:
            algorithm_data = self._get_hybrid_trend(mnemonic_id, in_time)
        else:
            algorithm_data = self.get_data_trend(mnemonic_id, in_time, DEFAULT)

        return algorithm_data

    def get_stat_list(self, mnemonic_id: str, state : Optional[str]) -> List[DataPoint]:
        """
        Retrieves historical statistics (mean, sigma, etc.) for a specific mnemonic and state.

        Args:
            mnemonic_id (str): ID of the mnemonic.
            state (Optional[str]): The operational state. Defaults to 'DEFAULT'.

        Returns:
            List[DataPoint]: The list of statistical data points.
        """
        if not state:
            state = DEFAULT
        stat_list = self.sdt_data_io.get_data_stats(mnemonic_id, data_buffer.session_type, data_buffer.session_end, state)
        return stat_list

    def get_data_trend(self, mnemonic_id: str, input_times: List[float], state: str = DEFAULT) -> Optional[AlgorithmData]:
        """
        Retrieves a single trained model for a specific state and time range.

        Args:
            mnemonic_id (str): ID of the mnemonic.
            input_times (List[float]): Reference time and window [ref, window].
            state (str): Operational state.

        Returns:
            Optional[AlgorithmData]: The trained model parameters, or None.
        """
        start_time = input_times[0] - input_times[1]
        end_time = input_times[0] + input_times[1]

        algorithm_data = self.sdt_data_io.get_data_trend(mnemonic_id, start_time, end_time, state)
        if not algorithm_data:
            return None
        else:
            return algorithm_data

    def _get_hybrid_trend(self, mnemonic_id: str, in_time: List[float]) -> Optional[HybridStateData]:
        """
        Retrieves and reconstructs a multi-state baseline trend for a hybrid model.

        Args:
            mnemonic_id (str): ID of the mnemonic.
            in_time (List[float]): Default input time window.

        Returns:
            Optional[HybridStateData]: The reconstructed multi-state data object.
        """
        prev_trend_container: HybridTrend = create_hybrid_trend(mnemonic_id, PREV)
        current_trend_container: HybridTrend = create_hybrid_trend(mnemonic_id, CURRENT)
        current_trends = current_trend_container.get_data_trends()
        hybrid_state_data = HybridStateData(
            mnemonic_id=mnemonic_id,
            alg_name=HYBRID,
            state_zones=None,
            data_trend_list=[]
        )
        for trend in current_trends:
            if trend:
                state = trend.get_state()
                _prev_trend = _get_prev_trend(prev_trend_container, trend)
                if _prev_trend:
                    ranges = _get_query_range(_prev_trend, in_time, state)
                    if ranges:
                        single_state_data = self.get_data_trend(mnemonic_id, ranges, state)
                        if single_state_data:
                            hybrid_state_data.data_trend_list.append(single_state_data)
        return hybrid_state_data

    def get_sigma(self, mnemonic_id: str, start_time: float, end_time: float, state: str) -> Optional[List[DataPoint]]:
        """
        Retrieves standard deviation history for monitoring purposes.

        Args:
            mnemonic_id (str): ID of the mnemonic.
            start_time (float): Start timestamp.
            end_time (float): End timestamp.
            state (str): Operational state.

        Returns:
            Optional[List[DataPoint]]: List of sigma data points.
        """
        return self.sdt_data_io.get_sigma(mnemonic_id, start_time, end_time, state)

    def close(self):
        """
        Closes the underlying storage plugin and releases resources.
        """
        if self.sdt_data_io:
            self.sdt_data_io.close()
