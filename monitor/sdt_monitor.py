import logging
import os
import sys
from typing import List, Optional

import numpy as np

from algorithm.hybrid.hybrid_trend import HybridTrend
from config.sdt_constants import ELIMIT

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import sdt_config
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_trend import DataTrend, WLIMIT
from algorithm.outlier import Outlier
from training.training_set import TrainingSet
from training.preprocessing.ex_zone import ExZone
from util.time_util import get_time_tag_from_seconds

# Constants
CONTEXT = "MONITOR"
ERROR = 1
WARNING = 0
INFO = "INFO"


def monitor(node: MnemonicNode, current_trend: DataTrend, training_set: TrainingSet, ex_zone: Optional[ExZone], pattern_times: Optional[np.ndarray] = None) -> bool:
    """
    Orchestrates the detection of outliers for a mnemonic node.
    
    It invokes the check_outlier logic and, if outliers are found, adds them 
    to the node's internal list and returns True.

    Args:
        node (MnemonicNode): The tree node representing the mnemonic.
        current_trend (DataTrend): The trained model to check against.
        training_set (TrainingSet): The set of data points to evaluate.
        ex_zone (Optional[ExZone]): Defines time regions to ignore (e.g., maneuvers).
        pattern_times (Optional[np.ndarray]): Unused in current implementation.

    Returns:
        bool: True if new outliers were detected, False otherwise.
    """
    record_list = check_outlier(current_trend, training_set, WARNING, ex_zone)
    if record_list:
        logging.info(f"{CONTEXT}: The number of outliers for {current_trend.mnemonic_id} is: {len(record_list)}")
        node.add_outliers(record_list)
        return True
    return False


def check_outlier(current_trend: DataTrend, training_set: TrainingSet, limit_index: int, ex_zone: Optional[ExZone]) -> Optional[List[Outlier]]:
    """
    Core outlier detection logic using vectorized NumPy operations for high performance.
    
    It compares the raw data against the model's upper and lower bounds. Values 
    exceeding these bounds, and not located within an exclusion zone, are 
    flagged as outliers.

    Args:
        current_trend (DataTrend): The reference model.
        training_set (TrainingSet): The data points and DQF flags.
        limit_index (int): Indicates which threshold to use (WARNING or ERROR).
        ex_zone (Optional[ExZone]): The exclusion zone filter.

    Returns:
        Optional[List[Outlier]]: A list of detected Outlier objects, or None if 
            the input is invalid.
    """
    inputs = training_set.inputs
    results = training_set.raw
    dqf = training_set.dqf

    if inputs.size == 0:
        return None

    alg = current_trend.get_algorithm()
    ocscale = float(alg.get_attribute("ocscale") or 1.0)
    sigma = current_trend.get_stddev()
    if sigma == 0 or ocscale == 0:
        return None

    limit_str = ELIMIT if limit_index == ERROR else WLIMIT

    # Bulk calculation of trend limits for all time points
    all_limits = current_trend.get_all_trend_values(inputs, limit_str)
    if all_limits is None: 
        return None

    l_high = all_limits[:, 0]
    l_low = all_limits[:, 1]

    # Filter out points in exclusion zones
    times = inputs[:, 0]
    if ex_zone is not None:
        in_zone_mask = ex_zone.is_in_zone(times)
    else:
        in_zone_mask = np.zeros(len(times), dtype=bool)

    # Detect points exceeding thresholds
    diffs = l_low - results
    thresholds = l_high - l_low
    abs_diffs = np.abs(diffs)

    outlier_mask = abs_diffs > thresholds
    # Ignore points in exclusion zones
    outlier_mask = outlier_mask & (~in_zone_mask)

    outlier_indices = np.where(outlier_mask)[0]
    if len(outlier_indices) == 0:
        return []

    # Mark DQF as 0 for outliers
    if isinstance(dqf, np.ndarray):
        dqf[outlier_indices] = 0

    # Calculate normalized differences
    o_diffs = diffs[outlier_indices]
    o_thresholds = thresholds[outlier_indices]
    normalized_diffs = (o_diffs - o_thresholds * np.sign(o_diffs)) / (sigma * ocscale)

    # Determine state names for each outlier
    if isinstance(current_trend, HybridTrend):
        h_trend : HybridTrend = current_trend
        state_indices = [h_trend.get_state_name(times[idx]) for idx in outlier_indices]
    else:
        state_indices = ["DEFAULT"] * len(outlier_indices)

    return [
        Outlier(
            mnemonic_id=current_trend.mnemonic_id,
            time=times[idx],
            t_string=get_time_tag_from_seconds(times[idx]),
            value=float(results[idx]),
            diff=float(normalized_diffs[i]),
            limit=float(o_thresholds[i]),
            state=state_indices[i],
            time_index=int(idx)
        )
        for i, idx in enumerate(outlier_indices)
    ]

def init_dqf(input_trend: DataTrend, training_set: TrainingSet, pattern_times: np.ndarray, ex_zone: Optional[ExZone]):
    """
    Initializes the Data Quality Flags (DQF) of a new training set.
    
    This is performed at the start of a session by checking new data against 
    the model from the previous session. If the outlier ratio is too high, 
    all DQF flags are reset to 1 (ignoring previous session's filter) to allow 
    the model to adapt to large shifts.

    Args:
        input_trend (DataTrend): The model from the previous session.
        training_set (TrainingSet): The data for the current session.
        pattern_times (np.ndarray): Unused.
        ex_zone (Optional[ExZone]): The exclusion zone filter.
    """
    outlier_ratio = input_trend.get_algorithm().get_attribute_double("OUTLIERRATIO")
    if outlier_ratio == 0.0:
        outlier_ratio = float(sdt_config.get_config_value("OUTLIERRATIO") or 0.0)

    outlier_list = check_outlier(input_trend, training_set, ERROR, ex_zone)
    training_set_sample_size = training_set.raw.size
    
    if outlier_ratio > 0 and outlier_list:
        ratio = len(outlier_list) / training_set_sample_size
        logging.info(f"{CONTEXT}: Outliers from initDQF for {input_trend.mnemonic_id}: {len(outlier_list)}, Ratio: {ratio:.4f}")
            
        if ratio > outlier_ratio:
            logging.info(f"{CONTEXT}: Outlier ratio exceeded ({ratio:.4f} > {outlier_ratio:.4f}). Resetting DQF.")
            training_set.dqf = np.ones(training_set_sample_size, dtype=np.int8)
