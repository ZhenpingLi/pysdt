import logging
import os
import sys
from typing import List, Tuple

import numpy as np

from algorithm.ridgefe.fourier_expansion_function import FourierExpansionFunction, HOUR_IN_SECONDS

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_trend import DataTrend
from algorithm.ridgeregression.ridge_reg_training_worker import RidgeRegTrainingWorker
CONTEXT: str = "RIDGEFEWorker"

class RidgeFETrainingWorker(RidgeRegTrainingWorker):
    """
    TrainingWorker implementation for Ridge Fourier Expansion (RidgeFE) models.
    
    This worker fits a periodic model to telemetry data using a Fourier 
    series expansion. It uses Ridge Regression (L2 regularization) to prevent 
    overfitting, which is common in high-order harmonic models.
    """

    def set_config(self, node: MnemonicNode):
        """
        Configures the worker with periodic parameters and Fourier order.

        Args:
            node (MnemonicNode): The mnemonic node to be trained.
        """
        super().set_config(node)
        
        self.num_training = self.algorithm.get_np() or 1
        dim = self.algorithm.get_dimension()
        # Parameter count = DC + 2 * order (sine and cosine pairs)
        self.param_size = 2 * dim[0] + 1
        
        num_pattern = len(self.pattern_times) // 2
        if num_pattern > self.num_pattern_in_training:
            self.num_pattern_in_training = num_pattern
            
        self.function = FourierExpansionFunction()
        self.function.set_dim([self.param_size])
        self.function.set_pattern_period(self.pattern_period)
        
        # Max input limit in normalized hours
        self.x_scale_limits = self.pattern_period / HOUR_IN_SECONDS
        
        lambda_str = self.algorithm.get_attribute("lambda")
        if lambda_str:
            self.lambda_val = float(lambda_str)
        
        self.retraining_sample = 3600

    def get_x_elements(self, inputs: np.ndarray, outputs: np.ndarray, current_trend: DataTrend) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms raw timestamps into a Fourier feature matrix.
        
        It first normalizes absolute time to cycle-relative time, sorts the 
        data, and then evaluates the Fourier basis functions (sin/cos) 
        for every point.

        Args:
            inputs (np.ndarray): Raw input telemetry matrix (time).
            outputs (np.ndarray): Observed target values.
            current_trend (DataTrend): Trend model used for time transformation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Fourier_feature_matrix, sorted_outputs).
        """
        # Map absolute Unix time to cycle-relative hours [0, T]
        model_times = np.array([current_trend.get_data_model_time(time[0]) for time in inputs])
        
        # Ensure data is sorted by relative time
        sort_indices = np.argsort(model_times)
        sort_model_times = model_times[sort_indices]
        outputs = outputs[sort_indices]
        
        # Filter for data within valid periodic boundaries
        mask = sort_model_times < self.x_scale_limits
        valid_model_times = sort_model_times[mask]
        valid_outputs = outputs[mask]
        
        # Evaluate sin/cos basis functions for each time point
        # Note: func(t)[1:] skips the DC term (added by the Ridge Regression engine)
        features = np.array([self.function.func(mt)[1:] for mt in valid_model_times])
        
        return features.astype(np.float64), valid_outputs.astype(np.float64)


def sync_drop_correlated(inputs_arr: np.ndarray, outputs_arr: np.ndarray, threshold=0.99) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifies and removes redundant, highly correlated feature columns.

    Args:
        inputs_arr (np.ndarray): The feature matrix.
        outputs_arr (np.ndarray): The target vector.
        threshold (float): Correlation threshold for dropping. Defaults to 0.99.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered feature matrix and outputs.
    """
    import pandas as pd
    # Calculate pairwise correlation matrix
    df_in = pd.DataFrame(inputs_arr)
    corr_matrix = df_in.corr().abs()

    # Identify upper triangle of the matrix to avoid self-correlation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify columns to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Keep only unique, non-correlated columns
    keep_indices = [i for i in range(inputs_arr.shape[1]) if i not in to_drop]
    filtered_inputs = inputs_arr[:, keep_indices]

    if len(to_drop) > 0:
        logging.info(f"{CONTEXT}: Dropped {len(to_drop)} redundant correlated columns.")
        
    return filtered_inputs, outputs_arr
