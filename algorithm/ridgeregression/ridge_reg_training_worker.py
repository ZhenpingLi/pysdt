import numpy as np
from typing import List, Optional, Tuple
import sys
import os
import logging
from abc import abstractmethod

from config.sdt_constants import HOUR_IN_SECONDS
from training.preprocessing.lttb_filter import LTTBFilter
from util import sdt_util

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config.sdt_config as sdt_config
from algorithm.data_trend import DataTrend
from training.training_set import TrainingSet
from training.training_worker import TrainingWorker
from algorithm.ridgeregression import ridge_regression as rr

# --- Constants ---
CONTEXT = "RidgeTrainingWorker"
LINEAR = 1


class RidgeRegTrainingWorker(TrainingWorker):
    """
    Abstract base class for workers employing Ridge Regression (L2 regularization).
    
    This class provides the common infrastructure for fitting linear models where 
    overfitting or matrix ill-conditioning is a concern. It handles:
    1. Data preparation: filtering outliers (via DQF) and downsampling large 
       datasets (via LTTB) for efficient fitting.
    2. Feature matrix construction: delegating basis function generation to subclasses.
    3. Regularized optimization: solving the ridge regression equation.
    """

    def __init__(self):
        """Initializes the RidgeRegTrainingWorker with regularization parameters."""
        super().__init__()
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.lambda_val: float = float(sdt_config.get_config_value("LAMBDA") or 0.01)
        self.param_size: int = 0
        self.retraining_sample = sdt_config.get_int_property("RETRAINSAMPLE") or 2400
        self.function = None # To be initialized by concrete subclasses (e.g., Fourier)
        logging.info(f"[{CONTEXT}] Initialized with Lambda: {self.lambda_val}")
        self.x_scale_limits = self.pattern_period / HOUR_IN_SECONDS

    def get_algorithm_type(self) -> int:
        """Returns the algorithm type code (LINEAR)."""
        return LINEAR

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Executes the ridge regression fitting process.
        
        It orchestrates the initialization of inputs, executes the solver, 
        and updates the model parameters.

        Args:
            training_set (TrainingSet): Telemetry data points.
            current_trend (DataTrend): Model object to populate with weights.
            input_trend (Optional[DataTrend]): Baseline model for incremental context.
            
        Raises:
            Exception: If no valid data is available after filtering.
        """
        init_params = input_trend.get_params() if input_trend else None
        
        # Configure basis function period if applicable
        if hasattr(self.function, 'set_pattern_period'):
            self.function.set_pattern_period(self.pattern_period)
        
        # Prepare the X (feature) and y (target) matrices
        self._init_lsf_input(training_set, init_params, current_trend)
        
        if self.x_data is None or self.y_data is None or self.x_data.size == 0:
            raise Exception(f"[{CONTEXT}] No valid data available for {current_trend.mnemonic_id}")

        logging.info(f"[{CONTEXT}] Running Ridge Regression fit for {current_trend.mnemonic_id}.")
        
        # Solve for coefficients: (X'X + λI)w = X'y
        params = rr.run_ridge_regression(self.x_data, self.y_data, self.lambda_val)
        
        current_trend.set_params(params.tolist() if isinstance(params, np.ndarray) else params)
        logging.info(f"[{CONTEXT}] Successfully trained {len(params)} parameters.")

    def _init_lsf_input(self, training_set: TrainingSet, init_params: Optional[List[float]], current_trend: DataTrend):
        """
        Prepares high-quality, manageable datasets for the regression solver.
        
        It performs the following:
        1. Filters out outliers based on DQF.
        2. If the dataset is too large (>15,000 points), it applies LTTB 
           downsampling to reduce size while preserving the trend's shape.
        3. Invokes the subclass-specific feature generation logic.
        """
        inputs = training_set.inputs
        outputs = training_set.raw
        dqf = training_set.dqf
        
        valid_mask = (dqf == 1)
        if not np.any(valid_mask):
            self.x_data = np.array([])
            self.y_data = np.array([])
            return

        # 1. Initial filter based on DQF
        if not np.all(valid_mask):
            v_inputs = inputs[valid_mask]
            v_outputs = outputs[valid_mask]
            filtered_set = TrainingSet(mnemonic_id=training_set.mnemonic_id, inputs=v_inputs,
                                       raw=v_outputs, outputs=v_outputs, 
                                       dqf=np.ones(len(v_inputs), dtype=np.int8))
        else:
            filtered_set = training_set

        # 2. Downsample if dataset is massive to prevent memory/CPU issues
        if len(filtered_set.outputs) > 15000:
            sdt_util.sort_training_set(filtered_set)
            mp_filter = LTTBFilter(self.retraining_sample)
            filtered_set = mp_filter.filter(filtered_set)
            logging.info(f"[{CONTEXT}] Large dataset detected. Downsampled to {len(filtered_set.outputs)} points.")

        # 3. Generate the final feature matrix
        self.x_data, self.y_data = self.get_x_elements(filtered_set.inputs, filtered_set.outputs, current_trend)

    @abstractmethod
    def get_x_elements(self, inputs: np.ndarray, outputs: np.ndarray, current_trend: DataTrend) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the feature matrix (X) from the input telemetry.
        
        Must be implemented by subclasses to define the model type 
        (e.g., polynomial powers or Fourier harmonics).

        Args:
            inputs (np.ndarray): Cleaned input feature matrix.
            outputs (np.ndarray): Cleaned target values.
            current_trend (DataTrend): Model context.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_matrix, y_vector).
        """
        pass
