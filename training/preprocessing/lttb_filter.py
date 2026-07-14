import numpy as np
from typing import Optional
import sys
import os
from lttb import downsample

from util import sdt_util

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.training_set import TrainingSet


class LTTBFilter:
    """
    Downsampling filter using the Largest-Triangle-Three-Buckets (LTTB) algorithm.
    
    LTTB is a shape-preserving downsampling technique that maintains the visual 
    characteristics of the original time series while significantly reducing 
    the number of data points. This is particularly useful for preparing 
    high-frequency telemetry for long-term trending or visualization.
    """

    def __init__(self, n_out: int):
        """
        Initializes the LTTBFilter.

        Args:
            n_out (int): The target number of data points to remain after 
                downsampling. Must be at least 2.

        Raises:
            ValueError: If n_out is less than 2.
        """
        if n_out < 2:
            raise ValueError("n_out must be at least 2 for LTTB algorithm.")
        self.n_out = n_out

    def filter(self, training_set: TrainingSet) -> Optional[TrainingSet]:
        """
        Applies the LTTB downsampling to the provided TrainingSet.
        
        The method uses the first column of the training set's inputs as the 
        X-axis (time) and the outputs as the Y-axis. It ensures data is sorted 
        by time before downsampling and preserves the Data Quality Flags (DQF) 
        for the selected points.

        Args:
            training_set (TrainingSet): The input telemetry dataset.

        Returns:
            Optional[TrainingSet]: A new, smaller TrainingSet containing the 
                downsampled points, or the original set if its size was 
                already below the target.
        """
        if len(training_set.outputs) <= self.n_out:
            return training_set

        sdt_util.sort_training_set(training_set)
        inputs = training_set.inputs
        outputs = training_set.outputs
        dqf = training_set.dqf

        if inputs is None or outputs is None:
            return None

        # Prepare 1D arrays for the algorithm
        times = inputs[:, 0].flatten()
        outputs_flat = outputs.flatten()
        
        # Step 1: Combine into (N, 2) array as required by the 'lttb' library
        data_to_downsample = np.column_stack((times, outputs_flat))
        
        # Step 2: Execute downsampling
        downsampled_data = downsample(data_to_downsample, n_out=self.n_out)
        
        # Step 3: Extract results
        downsampled_inputs = downsampled_data[:, 0].reshape(-1, 1)
        downsampled_outputs = downsampled_data[:, 1]
        
        # Step 4: Synchronize DQF mask for the selected timestamps
        original_indices = np.searchsorted(times, downsampled_inputs.flatten())
        downsampled_dqf = dqf[original_indices] if dqf is not None else None

        # Step 5: Construct the resulting TrainingSet
        return TrainingSet(mnemonic_id=training_set.mnemonic_id,
                           inputs=downsampled_inputs,
                           raw=downsampled_outputs,
                           outputs=downsampled_outputs,
                           dqf=downsampled_dqf)
