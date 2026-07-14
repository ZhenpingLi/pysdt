import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np

from algorithm.data_trend import DataTrend
from algorithm.ridgefe.ridge_fe_training_worker import RidgeFETrainingWorker
from algorithm.rmfe import fitting_function
from config.sdt_constants import DAY_IN_SECONDS
from training.training_set import TrainingSet

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.mnemonic_node import MnemonicNode
from training import data_buffer

# --- Constants ---
CONTEXT = "RMFETRAINING"
FACTOR = 14
NUMPATTERN = "numpattern"


class RidgeMFETrainingWorker(RidgeFETrainingWorker):
    """
    TrainingWorker implementation for Ridge Modified Fourier Expansion (RMFE).
    
    This worker specializes in modeling complex, high-dimensional periodic 
    behaviors by employing a modified Fourier expansion. It uses linear 
    least squares optimization (via np.linalg.lstsq) to solve for model 
    coefficients across extended pattern windows (e.g., 14-day cycles).
    """

    def __init__(self):
        """Initializes the RidgeMFETrainingWorker."""
        super().__init__()

    def set_config(self, node: MnemonicNode):
        """
        Configures the worker with RMFE-specific periodic parameters.

        Args:
            node (MnemonicNode): The node being trained.
        """
        super().set_config(node)

        # Retrieve cycle scaling factor from algorithm attributes
        factor_str = self.algorithm.get_attribute(NUMPATTERN)
        factor = float(factor_str) if factor_str else FACTOR
        
        model_time = data_buffer.get_session_model_time(self.algorithm)
        if not model_time:
             return
             
        self.num_training = 1
        self.num_pattern_in_training = 1
        
        # RMFE often operates on larger multi-day windows
        self.pattern_period = model_time.get_model_period() * factor
        self.ref_time = model_time.get_reference_time()
        self.x_scale_limits = self.pattern_period / DAY_IN_SECONDS
        
        # Define 2-cycle training window
        self.pattern_times = np.array([
            self.ref_time,
            self.ref_time + self.pattern_period,
            self.ref_time + 2 * self.pattern_period
        ])

        self.retraining_sample = 10000
        fitting_function.set_period(self.pattern_period)
        
        # Parse Fourier order and set basis function dimensionality
        dims = self.algorithm.get_dimension()
        order = dims[0] if dims else 0
        self.param_size = 2 * order + 1
        fitting_function.set_dim(self.param_size)

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Executes the RMFE model fitting process.
        
        Orchestrates feature matrix construction and solves the linear 
        least-squares problem.

        Args:
            training_set (TrainingSet): Telemetry data points.
            current_trend (DataTrend): Model object to update.
            input_trend (Optional[DataTrend]): Baseline model.
        """
        init_params = input_trend.get_params() if input_trend else None

        if hasattr(self.function, 'set_pattern_period'):
            self.function.set_pattern_period(self.pattern_period)

        # 1. Construct RMFE feature matrix (X) and target vector (y)
        self._init_lsf_input(training_set, init_params, current_trend)

        if self.x_data is None or self.y_data is None or self.x_data.size == 0:
            logging.error(f"{CONTEXT}: No valid data points found for {current_trend.mnemonic_id}.")
            return

        logging.info(f"[{CONTEXT}] Solving RMFE system for {current_trend.mnemonic_id}.")

        # 2. Optimized linear solve
        params = self.run_curve_fit(self.x_data, self.y_data)
        current_trend.set_params(params.tolist() if isinstance(params, np.ndarray) else params)

    def get_x_elements(self, inputs: np.ndarray, outputs: np.ndarray, current_trend: DataTrend) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms raw timestamps into high-dimensional RMFE features.

        Args:
            inputs (np.ndarray): Input feature matrix (time).
            outputs (np.ndarray): Observed values.
            current_trend (DataTrend): Trend model for time normalization.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Feature_matrix, output_vector).
        """
        # Map timestamps to normalized hours in cycle
        model_times = np.array([current_trend.get_data_model_time(t[0]) for t in inputs])
        
        # Generate basis function expansion for each point
        features = np.array([fitting_function.func(mt) for mt in model_times])
        
        return features.astype(np.float64), outputs.astype(np.float64)

    def run_curve_fit(self, x_train: np.ndarray, y_train: np.ndarray, init_params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Performs the linear least-squares optimization using SVD decomposition.

        Args:
            x_train (np.ndarray): The RMFE feature matrix.
            y_train (np.ndarray): The target values.
            init_params (Optional[np.ndarray]): Unused.

        Returns:
            np.ndarray: The resulting optimized model coefficients.
        """
        # Linear solve: Find w that minimizes ||Xw - y||^2
        a, residuals, rank, s = np.linalg.lstsq(x_train, y_train, rcond=None)
        
        logging.info(f"[{CONTEXT}] Optimization complete. Rank: {rank}, Residuals: {residuals}")
        return a
