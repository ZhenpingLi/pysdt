import logging
import os
import sys
from typing import List, Optional, Tuple, Any

import numpy as np

from algorithm.algorithm_def import AlgorithmDef
from algorithm.data_point import DataPoint
from algorithm.hybrid_state_data import HybridStateData
from algorithm.single_state_data import SingleStateData
from algorithm.training_output import TrainingOutputData
from algorithm.training_output_processor import is_overlap
from config.sdt_constants import NORMALSTATUS, ERRORSTATUS, WARNINGSTATUS, TRAININGERROR
from posttraining.clustering.sdt_event_data import SDTEventData
from posttraining.mnemonic_status import MnemonicStatus
from sdtdb import sdt_db
from training import data_buffer

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.sdt_config as sdt_config
import training.data_buffer as db
from algorithm.data_trend import HOUR_IN_SECONDS

# --- Constants ---
CONTEXT = "DataQualityMetrics"
DEFAULT = "default"
ECL = "ecl"
CHARGE = "charge"
HYBRID = "hybrid"
TPCTREND = "tpctrend"

STATUS_DIMENSION = 9
"""Dimensionality of the status metrics array."""

is_in_yaw_flip = False


class EvaluationAttributes:
    """
    Helper class to parse and store evaluation-specific attributes for an algorithm.
    
    It extracts parameters like operational scale and state-specific checks 
    (e.g., eclipse or yaw flip) from the algorithm definition.
    """
    KEYS: List[str] = ['ops_scale', 'check_ecl', 'tpctrend', 'training_type', 'check_yaw']

    def __init__(self, algorithm: AlgorithmDef):
        """
        Initializes EvaluationAttributes from an AlgorithmDef.

        Args:
            algorithm (AlgorithmDef): The algorithm definition to parse.
        """
        global is_in_yaw_flip
        ops_scale_str = algorithm.get_attribute("opsscale")
        self.ops_scale: float = float(ops_scale_str) if ops_scale_str else 1.0
        
        # Check global mission state for yaw flip
        is_in_yaw_flip = data_buffer.get_ex_zone("GNC_RO_YAW_FLIP_STATE", 
                                                data_buffer.session_time, 
                                                data_buffer.session_end) is not None
                                                
        tpc_trend_str = algorithm.get_attribute(TPCTREND)
        self.check_ecl: bool = tpc_trend_str == ECL if tpc_trend_str else False
        self.check_yaw: bool = tpc_trend_str == "yaw" if tpc_trend_str else False
        self.training_type = 1 if algorithm.is_derivative_check() else 0

    def __getitem__(self, item):
        """Allows dict-like access to attributes."""
        if item in EvaluationAttributes.KEYS:
            return getattr(self, item)
        else:
            raise KeyError(f"Invalid key: {item} for the EvaluationAttributes object")


def evaluate_quality_metrics(mnemonic_output: TrainingOutputData, is_in_noise: bool) -> Tuple[List[MnemonicStatus], List[SDTEventData]]:
    """
    Main entry point for calculating operational health status for a mnemonic.
    
    It evaluates multiple metrics (TPC, outlier values, statistics) and 
    determines the overall status level (NORMAL, WARNING, ERROR). It handles 
    both single-state and hybrid multi-state models.

    Args:
        mnemonic_output (TrainingOutputData): The direct output of a training session.
        is_in_noise (bool): Flag indicating if the mnemonic is considered noisy.

    Returns:
        Tuple[List[MnemonicStatus], List[SDTEventData]]: A tuple containing 
            the list of health status records and any newly generated TPC events.
    """
    algorithm: AlgorithmDef = AlgorithmDef(sdt_db.get_algorithm(mnemonic_id=mnemonic_output.mnemonic_id))
    evaluation_attributes = EvaluationAttributes(algorithm)
    cutoff_time = db.session_end - (db.get_training_delta() * HOUR_IN_SECONDS)
    
    event_data_list: List[SDTEventData] = mnemonic_output.mnemonic_event_list if mnemonic_output.mnemonic_event_list else []
    tpc_mnemonic_event_list: List[SDTEventData] = []
    status: List[MnemonicStatus] = []
    
    if mnemonic_output.training_error is not None:
        # Map training errors to a specific status report
        status_points = create_training_error_status()
        status = [MnemonicStatus(mnemonic_id=mnemonic_output.mnemonic_id, 
                                 time=p.time, status_array=p.data, 
                                 state_id=DEFAULT) for p in status_points]
    else:
        for algorithm_data in mnemonic_output.algorithm_data_list:
            if algorithm_data.alg_name == HYBRID:
                _status, _tpc_events = init_hybrid_metrics(algorithm_data, event_data_list, cutoff_time, is_in_noise, evaluation_attributes)
                status.extend(_status)
                tpc_mnemonic_event_list.extend(_tpc_events)
            else:
                _status, _tpc_events = init_standard_metrics(algorithm_data, event_data_list, cutoff_time, is_in_noise, evaluation_attributes)
                status.extend(_status)
                tpc_mnemonic_event_list.extend(_tpc_events)
                
    # Store results back in output object for persistence
    mnemonic_output.ops_status = [DataPoint(s.time, s.status_array) for s in status]
    return status, tpc_mnemonic_event_list

def create_training_error_status() -> List[DataPoint]:
    """
    Creates a status metric array representing a failed training session.

    Returns:
        List[DataPoint]: A list containing a single DataPoint marked with TRAININGERROR.
    """
    data = np.zeros(STATUS_DIMENSION)
    data[0] = TRAININGERROR
    return [DataPoint(time=db.session_time, data=data)]

def init_standard_metrics(trend: SingleStateData, mnemonics_event_list: List[SDTEventData], cutoff_time: float, is_in_noise: bool, evaluation_attributes: EvaluationAttributes) -> Tuple[List[MnemonicStatus], List[SDTEventData]]:
    """
    Calculates health metrics for a single-state model.
    
    It checks for temporal changes (TPC) and outlier values, adjusting thresholds 
    for special cases like eclipses or yaw flips.

    Args:
        trend (SingleStateData): The trained model data.
        mnemonics_event_list (List[SDTEventData]): Processed outlier events.
        cutoff_time (float): Time threshold for recent analysis.
        is_in_noise (bool): Noisy data indicator.
        evaluation_attributes (EvaluationAttributes): Configuration overrides.

    Returns:
        Tuple[List[MnemonicStatus], List[SDTEventData]]: The status and TPC events.
    """
    tpc = trend.tpc
        
    # Adjustment for TPC if already in a long event
    if tpc > sdt_config.TPCWARNING and db.is_in_the_long_event_list_in_prev_session(trend.mnemonic_id):
        tpc = sdt_config.TPCWARNING - 0.1
            
    # Adjustment for short eclipses
    if evaluation_attributes.check_ecl:
        ecl_dur = db.get_ecl_duration()
        if 1 < ecl_dur < sdt_config.ECLDUR and tpc > sdt_config.TPCWARNING:
            tpc = sdt_config.TPCWARNING - 0.1
                
    oc_value = get_event_value(mnemonics_event_list, cutoff_time, db.session_end, DEFAULT)
    
    # Adjustment for yaw flips
    if evaluation_attributes.check_yaw and is_in_yaw_flip:
        tpc = min(tpc, sdt_config.TPCWARNING - 0.1) if tpc >= sdt_config.TPCWARNING else tpc
        oc_value = min(oc_value, sdt_config.OCWARNING - 0.1) if oc_value > sdt_config.OCWARNING else oc_value
        
    data = np.zeros(STATUS_DIMENSION)
    data[1] = tpc
    data[2] = oc_value
    data[3] = evaluation_attributes.training_type
    data[0] = evaluate_ops_status(data, is_in_noise, evaluation_attributes.ops_scale)
    data[4] = trend.sigma
    
    if trend.stat_list and trend.stat_list[-1].data is not None:
        data[5:9] = trend.stat_list[-1].data
        
    tpc_event = update_tpc_event(data, trend, DEFAULT)
    status = [MnemonicStatus(mnemonic_id=trend.mnemonic_id, time=trend.ref_time, status_array=data, state_id=DEFAULT)]
    
    tpc_mnemonic_event_list = [tpc_event] if tpc_event else []
    return status, tpc_mnemonic_event_list


def init_hybrid_metrics(h_trend: HybridStateData, event_data_list: List[SDTEventData], cutoff_time: float, is_in_noise: bool, evaluation_attributes: EvaluationAttributes) -> Tuple[List[MnemonicStatus], List[SDTEventData]]:
    """
    Calculates health metrics for a multi-state (Hybrid) model.
    
    Evaluates each active state separately and aggregates the results.

    Args:
        h_trend (HybridStateData): The multi-state model data.
        event_data_list (List[SDTEventData]): Processed events.
        cutoff_time (float): Recent analysis threshold.
        is_in_noise (bool): Noisy data indicator.
        evaluation_attributes (EvaluationAttributes): Configuration overrides.

    Returns:
        Tuple[List[MnemonicStatus], List[SDTEventData]]: The list of status 
            records and TPC events for each state.
    """
    is_in_ecl = db.get_ecl_duration() > 100
    active_trends = [t for t in h_trend.data_trend_list if t is not None and (t.ref_time > cutoff_time or t.state == CHARGE or t.state == DEFAULT)]

    status_list: List[MnemonicStatus] = []
    tpc_event_list: List[SDTEventData] = []
    
    for trend_data in active_trends:
        try:
            state = trend_data.state
            tpc = trend_data.tpc
            oc_value = get_event_value(event_data_list, cutoff_time, db.session_end, state)
            
            if state in [DEFAULT, ECL] and is_in_ecl:
                ecl_dur = db.get_ecl_duration()
                if 0 < ecl_dur < sdt_config.ECLDUR and tpc > sdt_config.TPCWARNING:
                    tpc = sdt_config.TPCWARNING - 0.1
            elif state != DEFAULT:
                tpc /= 2.0
                
            if evaluation_attributes.check_yaw and is_in_yaw_flip:
                tpc = min(tpc, sdt_config.TPCWARNING - 0.1) if tpc >= sdt_config.TPCWARNING else tpc
                oc_value = min(oc_value, sdt_config.OCWARNING - 0.1) if oc_value > sdt_config.OCWARNING else oc_value
                
            stat_list = trend_data.stat_list
            data_point = stat_list[-1] if stat_list else None
            
            data = np.zeros(STATUS_DIMENSION)
            data[1] = tpc
            data[2] = oc_value
            data[3] = evaluation_attributes.training_type
            data[0] = evaluate_ops_status(data, is_in_noise, evaluation_attributes.ops_scale)
            data[4] = trend_data.sigma
            
            if data_point:
                data[5:9] = data_point.data
                
            ref_time = trend_data.ref_time if state != DEFAULT else db.session_time
            
            mnemonic_event = update_tpc_event(data, trend_data, state)
            if mnemonic_event:
                tpc_event_list.append(mnemonic_event)
                
            status_list.append(MnemonicStatus(mnemonic_id=trend_data.mnemonic_id, time=ref_time, status_array=data, state_id=state))
        except Exception as e:
            logging.error(f"Error in init_hybrid_metrics for {trend_data.mnemonic_id}: {e}")

    return status_list, tpc_event_list

def update_tpc_event(data: np.ndarray, data_trend: SingleStateData, state: str) -> Optional[SDTEventData]:
    """
    Creates a new TPC event if the temporal change exceeds thresholds.

    Args:
        data (np.ndarray): The status metric array.
        data_trend (SingleStateData): The trend being evaluated.
        state (str): The operational state name.

    Returns:
        Optional[SDTEventData]: A new TPC event object, or None.
    """
    if data[0] > NORMALSTATUS and data[1] > sdt_config.TPCWARNING:
        return SDTEventData(
            name=data_trend.mnemonic_id,
            event_value=float(data[1]),
            event_start=data_buffer.session_time,
            event_end=data_buffer.session_end,
            event_type="tpc",
            event_state=state
        )
    return None


def evaluate_ops_status(data: np.ndarray, is_in_noise: bool, ops_scale: float) -> int:
    """
    Heuristic to determine the aggregate operational status level.
    
    Compares the calculated metrics (TPC, Outliers) against warning and error 
    limits defined in the system configuration.

    Args:
        data (np.ndarray): The calculated health metrics.
        is_in_noise (bool): If True, ignores outlier-based errors.
        ops_scale (float): A scaling factor for the metrics.

    Returns:
        int: The status code (NORMALSTATUS, WARNINGSTATUS, or ERRORSTATUS).
    """
    state_status = NORMALSTATUS
    end_idx = 1 if is_in_noise else 2
    for j in range(1, end_idx):
        scaled_value = data[j] / ops_scale
        if scaled_value > sdt_config.error_limits[j - 1]:
            return ERRORSTATUS
        elif scaled_value > sdt_config.warning_limits[j - 1]:
            state_status = max(state_status, WARNINGSTATUS)
    return state_status

def get_event_value(event_list: List[SDTEventData], start: float, end: float, state: str) -> float:
    """
    Sums the values of all events that overlap with a time range and match a state.

    Args:
        event_list (List[SDTEventData]): The source event list.
        start (float): The start timestamp.
        end (float): The end timestamp.
        state (str): The target operational state.

    Returns:
        float: The aggregate event value.
    """
    if not event_list:
        return 0.0
    
    event_value = 0.0
    for event_data in event_list:
        if is_overlap(event_data, start, end) and event_data.event_state == state:
            event_value += event_data.event_value
    return event_value
