import logging
import os
import sys
from typing import Optional

import numpy as np

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sdtdb import sdt_db
from algorithm.algorithm_def import AlgorithmDef, DERIVATIVETYPE, MIXEDTYPE
from algorithm.mnemonic_node import MnemonicNode
from training.training_set import TrainingSet
from training.preprocessing.pre_processing import PreProcessing
from training.preprocessing.data_model_utility import get_derivative_data, get_mixed_type_data


# --- Constants ---
CONTEXT = "SHORTTERMPROCESSING"


def initialize_training_set_inputs(training_set: TrainingSet, alg: AlgorithmDef) -> Optional[TrainingSet]:
    """
    Initializes and cleans the input features and raw values of a TrainingSet.
    
    This function performs initial filtering for finite values (removing NaNs/Infs) 
    and handles specialized input transformations like derivative or mixed-type 
    input generation based on the algorithm configuration.

    Args:
        training_set (TrainingSet): The dataset to initialize.
        alg (AlgorithmDef): The algorithm definition used to determine 
            the transformation type.

    Returns:
        Optional[TrainingSet]: The initialized training set, or None if inputs 
            are empty.
    """
    mnemonic_id = training_set.mnemonic_id
    inputs = training_set.inputs
    if inputs is None or inputs.size == 0:
        logging.warning(f"{CONTEXT}: Input data list is empty for {mnemonic_id}")
        return None
        
    # Remove any non-finite values (NaN, Inf) from the raw data
    mask = np.isfinite(training_set.raw)
    training_set.inputs = inputs[mask]
    training_set.raw = training_set.raw[mask]
    training_set.dqf = training_set.dqf[mask]
    
    if training_set.outputs is not None:
        training_set.outputs = training_set.outputs[mask]
        
    frequency = sdt_db.get_mnemonic_type(mnemonic_id).frequency
    
    # Apply specialized feature transformations if multiple dimensions are present
    if inputs.shape[0] > 1:
        input_processing_type = alg.get_input_processing_type()
        if input_processing_type == DERIVATIVETYPE:
            input_data, output_data = get_derivative_data(training_set, alg.get_max_diff(), frequency)
            training_set = TrainingSet(mnemonic_id=mnemonic_id, inputs=input_data, 
                                     raw=output_data, outputs=output_data, 
                                     dqf=np.ones(len(input_data), dtype=np.int8))
        elif input_processing_type == MIXEDTYPE:
            input_data, output_data = get_mixed_type_data(training_set, alg.get_max_diff(), frequency)
            training_set = TrainingSet(mnemonic_id=mnemonic_id, inputs=input_data, 
                                     raw=output_data, outputs=output_data, 
                                     dqf=np.ones(len(input_data), dtype=np.int8))
                                     
    return training_set


def get_delta_set(input_set: TrainingSet, alg: AlgorithmDef, mnemonic_id: str) -> Optional[TrainingSet]:
    """
    Transforms a training set by calculating the first-order derivative of the outputs.
    
    The new output values are defined as (Δvalue / Δtime). This is used for 
    algorithms that model rates of change rather than absolute telemetry values.

    Args:
        input_set (TrainingSet): The source dataset.
        alg (AlgorithmDef): The algorithm definition containing 'scale' attribute.
        mnemonic_id (str): Mnemonic ID.

    Returns:
        Optional[TrainingSet]: A new TrainingSet containing the derivative data, 
            or None if insufficient data points.
    """
    inputs = input_set.inputs
    outputs = input_set.raw
    
    if inputs is None or len(inputs) < 2:
        return None

    time_list = inputs[:, 0]
    scale_s = alg.get_attribute("scale")
    scale = float(scale_s) if scale_s else float("inf")

    # Calculate differences between consecutive points
    delta_outputs = np.diff(outputs, prepend=outputs[0])
    delta_times = np.diff(time_list, prepend=time_list[0])

    # Filter out zeros (same-time points) and values exceeding the noise scale
    mask = (np.abs(delta_outputs) < scale) & (delta_times != 0)

    filtered_delta_times = delta_times[mask]
    filtered_delta_outputs = delta_outputs[mask]
    new_time_inputs = time_list[mask]
    
    derivatives = filtered_delta_outputs / filtered_delta_times
    
    output_set = TrainingSet(mnemonic_id=mnemonic_id,
                            inputs=new_time_inputs.reshape(-1, 1),
                            raw=derivatives,
                            outputs=derivatives,
                            dqf=np.ones(len(derivatives), dtype=np.int8))
                            
    output_set.stat_list = input_set.stat_list
    if input_set.coef_list is not None:
        output_set.coef_list = input_set.coef_list

    return output_set


class ShortTermProcessing(PreProcessing):
    """
    Preprocessing strategy for short-term (typically orbital or daily) training.
    
    This class specializes the PreProcessing framework to handle the nuances 
    of operational telemetry data, including high-frequency sampling, noise 
    filtering, and support for derivative-based trending.
    """

    def __init__(self):
        """Initializes the ShortTermProcessing instance."""
        super().__init__()

    def preprocess_training_set(self, training_set: TrainingSet, node: MnemonicNode, pattern_times: np.ndarray, pattern_period: float, pattern_offset: int) -> TrainingSet:
        """
        Main orchestration entry point for short-term preprocessing.

        Args:
            training_set (TrainingSet): Raw input data.
            node (MnemonicNode): Mnemonic identifier and hierarchy context.
            pattern_times (np.ndarray): Nominal start times for cycles.
            pattern_period (float): Nominal cycle duration.
            pattern_offset (int): Starting cycle index.

        Returns:
            TrainingSet: The fully processed and transformed dataset.
        """
        alg = AlgorithmDef(sdt_db.get_algorithm(training_set.mnemonic_id))
        initialize_training_set_inputs(training_set, alg)
        return self._get_training_set_detailed(training_set, pattern_times, pattern_period, pattern_offset)

    def _get_training_set_detailed(self, training_set: TrainingSet, ptimes: np.ndarray, period: float, pattern_offset: int) -> TrainingSet:
        """
        Executes the detailed preprocessing sequence: statistics calculation, 
        differentiation (if requested), and output mapping.
        """
        mnemonic_id = training_set.mnemonic_id
        alg = AlgorithmDef(sdt_db.get_algorithm(mnemonic_id))

        mn_type = sdt_db.get_mnemonic_type(mnemonic_id)
        sample_period = mn_type.frequency if mn_type else 1.0
        
        # 1. Calculate per-pattern cycle statistics
        self.get_stats(training_set, ptimes, sample_period, mnemonic_id, alg, period)

        # 2. Perform delta transformation if algorithm is derivative-based
        if alg.is_derivative_check():
            training_set = get_delta_set(training_set, alg, mnemonic_id)

        # 3. Synchronize raw and processed outputs for fitting
        training_set.outputs = training_set.raw

        return training_set
