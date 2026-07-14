import logging
import os
import sys
from typing import Tuple, Optional, List

import numpy as np

from training.training_set import TrainingSet

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.ridgeregression.ridge_reg_training_worker import RidgeRegTrainingWorker
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_trend import DataTrend
from algorithm.xfe.efe_function import EFEFunction
import training.data_buffer as db

# --- Constants ---
HOUR_IN_SECONDS = 3600
CONTEXT = "RIDGFE"


class RidgeXFETrainingWorker(RidgeRegTrainingWorker):
    """
    TrainingWorker implementation for the Ridge Extended Fourier Expansion (RidgeXFE).
    
    This worker fits high-complexity periodic models using an extended Fourier 
    expansion. It uses linear least squares optimization (np.linalg.lstsq) to 
    solve for model coefficients. It integrates EFEFunction to capture 
    multi-harmonic behaviors that may vary or exhibit drift over time.
    """

    def set_config(self, node: MnemonicNode):
        """
        Configures the worker with periodic and dimensionality parameters.

        Args:
            node (MnemonicNode): The node being trained.
        """
        super().set_config(node)
        
        model_time = db.get_session_model_time(self.algorithm)
        if not model_time:
             return
             
        self.ref_time = model_time.get_reference_time()
        self.pattern_times = model_time.get_pattern_times()
        self.num_pattern_in_training = len(self.pattern_times) if self.pattern_times is not None else 0
        self.pattern_period = model_time.get_model_period()
        
        # dimensionality: [order_sine, order_cosine] or similar based on EFEFunction
        dims = self.algorithm.get_dimension()
        if dims:
             self.param_size = 2 * (dims[0] + dims[1]) + 2
        
        self.function = EFEFunction()
        self.function.set_dim(dims)
        self.function.set_pattern_period(self.pattern_period)

    def get_x_elements(self, inputs: np.ndarray, outputs: np.ndarray, current_trend: DataTrend) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the feature matrix (X) using the EFE basis functions.

        Args:
            inputs (np.ndarray): Input feature matrix containing timestamps.
            outputs (np.ndarray): Observed target values.
            current_trend (DataTrend): Model context.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Feature_matrix, output_vector).
        """
        # Transform timestamps to normalized hours relative to ref_time
        features = np.array([self.function.func((t[0] - self.ref_time) / HOUR_IN_SECONDS) for t in inputs])
        
        return features.astype(np.float64), outputs.astype(np.float64)

    def run_curve_fit(self, x_train: np.ndarray, y_train: np.ndarray, init_params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solves the linear least-squares problem using singular value decomposition.

        Args:
            x_train (np.ndarray): The basis function feature matrix.
            y_train (np.ndarray): The target vector.
            init_params (Optional[np.ndarray]): Unused.

        Returns:
            np.ndarray: Optimized coefficients.
        """
        # (X'X)w = X'y solve via SVD
        a, residuals, rank, s = np.linalg.lstsq(x_train, y_train, rcond=None)
        
        logging.debug(f"[{CONTEXT}] Optimization complete. Rank: {rank}, Residuals: {residuals}")
        return a

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Executes the full RidgeXFE training cycle.

        Args:
            training_set (TrainingSet): Training telemetry.
            current_trend (DataTrend): Model object to populate.
            input_trend (Optional[DataTrend]): Baseline model.
        """
        init_params = input_trend.get_params() if input_trend else None

        if hasattr(self.function, 'set_pattern_period'):
            self.function.set_pattern_period(self.pattern_period)

        # 1. Prepare inputs (data filtering and downsampling handled by superclass)
        self._init_lsf_input(training_set, init_params, current_trend)

        if self.x_data is None or self.y_data is None or self.x_data.size == 0:
            logging.error(f"{CONTEXT}: No valid training data for {current_trend.mnemonic_id}.")
            return

        logging.info(f"[{CONTEXT}] Running high-dimensional periodic fit for {current_trend.mnemonic_id}.")

        # 2. Solve for coefficients
        params = self.run_curve_fit(self.x_data, self.y_data)
        
        current_trend.set_params(params.tolist() if isinstance(params, np.ndarray) else params)
        logging.info(f"[{CONTEXT}] Optimized {len(params)} XFE coefficients.")
