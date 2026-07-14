import logging
import os
import sys

import numpy as np

from sdtdb import sdt_db
from sdtdb.sdt_db import MIXEDTYPE
from training.training_set import TrainingSet

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_def import AlgorithmDef, NORMALTYPE, DERIVATIVETYPE


def get_derivative_data(training_set: TrainingSet, max_diff: float, frequency: float):
    """
    Transforms the training set inputs into a derivative-based feature set.
    
    It calculates the finite difference (Δvalue / Δtime) between rows in the 
    input matrix. It uses a centralized difference approach (i - (i-2)) to 
    provide more stable gradient features. It filters out any segments with 
    large data gaps or sudden spikes (max_diff).

    Args:
        training_set (TrainingSet): The dataset to transform. Modified in-place.
        max_diff (float): The maximum allowed difference between consecutive 
            points to consider the data valid.
        frequency (float): The nominal data sampling period, used to detect gaps.
    """
    input_array = training_set.inputs
    if input_array.shape[0] < 3:
        return

    # 1. Vectorized differences: row(i) - row(i-2)
    time_diffs = input_array[2:, 0] - input_array[:-2, 0]
    value_diffs_all = input_array[2:, 1:] - input_array[:-2, 1:]

    # 2. Identify valid segments (small gaps and smooth transitions)
    valid_mask = (np.all(np.abs(value_diffs_all) <= float(max_diff), axis=1)) & \
                 (time_diffs <= 3 * frequency)

    # 3. Extract and normalize
    valid_times = input_array[2:, 0][valid_mask]
    valid_diffs = value_diffs_all[valid_mask]
    valid_time_diffs = time_diffs[valid_mask][:, np.newaxis] 

    # 4. Update the TrainingSet in-place
    two_dim_array = np.column_stack((valid_times, valid_diffs / valid_time_diffs))
    output_array_filtered = training_set.raw[1:-1][valid_mask]
    
    training_set.inputs = two_dim_array
    training_set.raw = output_array_filtered
    training_set.outputs = output_array_filtered
    training_set.dqf = np.ones(len(valid_times), dtype=np.int8)


def get_mixed_type_data(training_set: TrainingSet, max_diff: float, frequency: float):
    """
    Creates a mixed feature set containing both original values and derivatives.
    
    The resulting input matrix will have the original features followed by 
    the calculated derivative features (Δvalue / Δtime). This allows models 
    to learn from both absolute levels and rates of change simultaneously.

    Args:
        training_set (TrainingSet): The dataset to transform. Modified in-place.
        max_diff (float): Threshold for outlier filtering.
        frequency (float): Nominal sampling frequency for gap detection.
    """
    input_array = training_set.inputs
    output_array = training_set.raw
    
    if input_array.shape[0] < 3:
        return

    # 1. Calculate derivatives using vectorized operations
    time_diffs = input_array[2:, 0] - input_array[:-2, 0]
    value_diffs_all = input_array[2:, 1:] - input_array[:-2, 1:]
    
    valid_mask = (np.all(np.abs(value_diffs_all) <= float(max_diff), axis=1)) & \
                 (time_diffs <= 3 * frequency)

    # 2. Gather original inputs and normalized derivatives
    valid_inputs = input_array[2:, :][valid_mask]
    valid_time_diffs = time_diffs[valid_mask][:, np.newaxis]
    valid_diffs = value_diffs_all[valid_mask]
    
    normalised_derivatives = valid_diffs / valid_time_diffs
    
    # 3. Combine original and derivative features
    two_dim_array = np.column_stack((valid_inputs, normalised_derivatives))
    output_array_filtered = output_array[1:-1][valid_mask]
    
    training_set.inputs = two_dim_array
    training_set.raw = output_array_filtered
    training_set.outputs = output_array_filtered
    training_set.dqf = np.ones(len(valid_inputs), dtype=np.int8)


def get_model_inputs(training_set: TrainingSet, algorithm: AlgorithmDef, mnemonic_id: str):
    """
    High-level dispatcher that applies the correct input transformation 
    to the TrainingSet based on algorithm configuration.

    Args:
        training_set (TrainingSet): The source dataset.
        algorithm (AlgorithmDef): The algorithm definition and configuration.
        mnemonic_id (str): The identifier for the mnemonic.
    """
    mn_type = sdt_db.get_mnemonic_type(mnemonic_id)
    if mn_type is not None:
        frequency = mn_type.frequency
        dim = len(training_set.inputs)
        
        if dim >= 1:
            input_processing_type = algorithm.get_input_processing_type()
            
            if input_processing_type == NORMALTYPE:
                if algorithm.is_derivative:
                    get_delta_set(training_set, algorithm, frequency)
            elif input_processing_type == DERIVATIVETYPE:
                get_derivative_data(training_set, algorithm.get_max_diff(), frequency)
            elif input_processing_type == MIXEDTYPE:
                get_mixed_type_data(training_set, algorithm.get_max_diff(), frequency)
            else:
                logging.warning(f"MODELINPUT: Type {input_processing_type} not defined for dim={dim}")


def get_delta_set(training_set: TrainingSet, algorithm: AlgorithmDef, frequency: float):
    """
    Calculates simple first-order derivatives (Δv/Δt) for the TrainingSet.
    
    This is a simplified version of get_derivative_data used for basic 
    one-dimensional derivative models.

    Args:
        training_set (TrainingSet): The dataset to modify.
        algorithm (AlgorithmDef): The algorithm configuration.
        frequency (float): Data sampling frequency.

    Returns:
        TrainingSet: The modified training set.
    """
    max_diff = float(algorithm.get_attribute_double("maxdiff"))

    # 1. Calculate time and value differences
    time_diffs = np.diff(training_set.inputs[:, 0])
    value_diffs = np.diff(training_set.raw)

    # 2. Filter for smooth data and small gaps
    valid_mask = (np.abs(value_diffs) <= max_diff) & (time_diffs <= 3 * frequency)

    # 3. Extract results
    valid_times = training_set.inputs[1:][valid_mask]
    dt = time_diffs[valid_mask]
    dv = value_diffs[valid_mask]

    # 4. Normalize differences to time (dV / dt)
    derivative_values = np.divide(dv, dt, out=np.zeros_like(dv), where=dt!=0)
    
    training_set.dqf = np.ones(len(valid_times), dtype=np.int8)
    training_set.inputs = valid_times
    training_set.outputs = derivative_values
    
    return training_set
