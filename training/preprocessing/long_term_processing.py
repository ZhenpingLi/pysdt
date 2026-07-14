import numpy as np
from typing import List, Optional
import sys
import os

from config.sdt_constants import LONGTERM
from sdtdb import sdt_db
from util import sdt_util

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.data_point import DataPoint
from algorithm.algorithm_def import AlgorithmDef
from algorithm.mnemonic_node import MnemonicNode
from training.training_set import TrainingSet
from training.preprocessing.pre_processing import PreProcessing
from .data_stat import DataStat

# --- Constants ---
MAXLIMIT = "maxlimit"
MINLIMIT = "minlimit"


def _get_stats_internal(training_set: TrainingSet, pattern_times: np.ndarray, pattern_period: float):
    """
    Calculates descriptive statistics for long-term datasets across pattern cycles.
    
    This function iterates through the nominal cycles (typically yearly or 
    seasonal for long-term) and populates the TrainingSet's stat_list.

    Args:
        training_set (TrainingSet): The dataset to process.
        pattern_times (np.ndarray): Nominal start times for pattern cycles.
        pattern_period (float): The duration of a single pattern period.
    """
    cal_stat = DataStat()
    num_patterns = len(pattern_times) - 1
    stat_list: List[Optional[DataPoint]] = [None] * num_patterns

    for i in range(num_patterns):
        start_time = float(pattern_times[i])
        end_time = float(pattern_times[i+1])

        sub_set = sdt_util.get_subset_by_time(training_set, start_time, end_time)
        if sub_set:
            stat_list[i] = cal_stat.get_stat(sub_set, start_time, None)

    training_set.stat_list = stat_list


def _init_dqf(training_set: TrainingSet, alg: AlgorithmDef):
    """
    Initializes Data Quality Flags (DQF) based on static absolute limits.
    
    This vectorized operation marks any data points exceeding configured 
    'maxlimit' or 'minlimit' as invalid (DQF=0).

    Args:
        training_set (TrainingSet): The dataset whose DQFs will be updated.
        alg (AlgorithmDef): The algorithm definition containing limit attributes.
    """
    max_limit = alg.get_attribute_double(MAXLIMIT)
    min_limit = alg.get_attribute_double(MINLIMIT)

    max_set = abs(max_limit) > 0
    min_set = abs(min_limit) > 0

    output = training_set.raw
    dqf = training_set.dqf

    # Create a mask for invalid points using NumPy vectorization
    bad_mask = np.zeros(len(output), dtype=bool)

    if max_set:
        bad_mask |= (output > max_limit)
    if min_set:
        bad_mask |= (output < min_limit)

    dqf[bad_mask] = 0
    training_set.dqf = dqf


class LongTermProcessing(PreProcessing):
    """
    Preprocessing strategy for long-term (multi-year) telemetry training.
    
    This class handles data that is often already aggregated or covers vast 
    time ranges. It focuses on coarse outlier removal based on static limits 
    and session-wide normalization.
    """

    def preprocess_training_set(self, training_set: TrainingSet, node: MnemonicNode, pattern_times: np.ndarray, pattern_period: float, pattern_offset: int) -> TrainingSet:
        """
        Main orchestration entry point for long-term data preparation.

        Args:
            training_set (TrainingSet): Raw input data.
            node (MnemonicNode): Mnemonic hierarchy context.
            pattern_times (np.ndarray): Nomimal start times for cycles.
            pattern_period (float): Nominal cycle duration.
            pattern_offset (int): Starting index for pattern cycles.

        Returns:
            TrainingSet: The fully processed dataset.
        """
        alg = AlgorithmDef(sdt_db.get_algorithm(mnemonic_id=node.name, training_type=LONGTERM))

        # 1. Calculate historical statistics
        _get_stats_internal(training_set, pattern_times, pattern_period)

        # 2. Perform normalization if required by the algorithm
        if alg and alg.is_normalized_check():
            from training.preprocessing.orbit_based_transform import OrbitBasedTransform
            transform = OrbitBasedTransform(alg, node.name)
            transform.transform(training_set, pattern_offset)
        else:
            training_set.outputs = training_set.raw
            
        return training_set

    @staticmethod
    def _get_stat_index(lg_path: List[str]) -> int:
        """
        Maps a logical path postfix to its corresponding statistical index.

        Args:
            lg_path (List[str]): The logic path components.

        Returns:
            int: The index (0-3) or -1 if invalid.
        """
        if len(lg_path) == 4:
            stat_map = {"mean": 0, "max": 1, "min": 2, "sigma": 3}
            return stat_map.get(lg_path[3], -1)
        return -1
