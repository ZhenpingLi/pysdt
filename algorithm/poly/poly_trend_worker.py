import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from algorithm.algorithm_def import AlgorithmDef
from algorithm.poly.poly_trend import PolyTrend
from sdtdb import sdt_db

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.ridgeregression.ridge_reg_training_worker import RidgeRegTrainingWorker
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_trend import DataTrend
from training.training_set import TrainingSet
import training.data_buffer

# --- Constants ---
CONTEXT = "PolyTrendWorker"
LINEAR = 1
MEAN = "mean"
PARAMS = "params"
LONGTERM = 1

class PolyTrendWorker(RidgeRegTrainingWorker):
    """
    TrainingWorker implementation for Polynomial Trend models.
    
    This worker fits a polynomial function (typically linear or low-order) 
    to telemetry data to model long-term drift. It uses robust non-linear 
    least squares optimization with a 'soft_l1' loss function to minimize 
     the impact of outliers during the fitting process.
    """

    def __init__(self):
        """Initializes the PolyTrendWorker."""
        super().__init__()
        self.m: int = 0
        self.s_index: int = 0
        self.checkdqf: bool = False
        self.params: Optional[np.ndarray] = None

    def set_config(self, node: MnemonicNode):
        """
        Configures the worker with parameters from the algorithm definition.
        
        It determines the polynomial degree (m) and starting index (s_index) 
        from dimensionality attributes and sets up the temporal context.

        Args:
            node (MnemonicNode): The mnemonic node to be trained.
        """
        self.mnemonic_id = node.name
        self.algorithm = AlgorithmDef(sdt_db.get_algorithm(self.mnemonic_id))
        self.algorithm_type = self.get_algorithm_type()
        
        model_time = training.data_buffer.get_session_model_time(self.algorithm)
        if not model_time:
             return
             
        self.pattern_times = model_time.get_pattern_times()
        self.pattern_period = model_time.get_model_period()
        self.ref_time = training.data_buffer.session_start
        
        if self.algorithm.get_np() > 0:
            self.num_pattern_in_training = self.algorithm.get_np()
        else:
            self.num_pattern_in_training = 1
            
        self.num_training = 1
        
        # Parse dimensionality: [s_index, m] or just [m]
        dim = self.algorithm.get_dimension()
        if len(dim) == 1:
            self.s_index = 0
            self.m = dim[0]
        elif len(dim) >= 2:
            self.s_index = dim[0]
            self.m = dim[1]
            
        self.param_size = self.m
        self.params = None
        
        param_string = self.algorithm.get_attribute(PARAMS)
        if param_string:
            self.params = np.array([float(p) for p in param_string.split('|')])
            
        self.checkdqf = self.algorithm.check_dqf()
        
        # Setup pattern times (start, mid, end) for a 48h training window
        self.pattern_times = np.array([
            training.data_buffer.session_end - 2 * self.num_pattern_in_training * self.pattern_period,
            training.data_buffer.session_end - self.num_pattern_in_training * self.pattern_period,
            training.data_buffer.session_end
        ])

    def get_algorithm_type(self) -> int:
        """Returns the algorithm type code (LINEAR)."""
        return LINEAR

    def get_x_elements(self, inputs: np.ndarray, outputs: np.ndarray, current_trend: DataTrend) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the polynomial feature matrix.

        Args:
            inputs (np.ndarray): Raw input telemetry matrix.
            outputs (np.ndarray): Target values.
            current_trend (DataTrend): The model object used for feature construction.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (feature_matrix, outputs).
        """
        poly_trend : PolyTrend = current_trend
        x_elements = np.array([poly_trend.poly_function(i) for i in inputs])
        return x_elements, outputs

    def _init_lsf_input(self, training_set: TrainingSet, init_params: Optional[List[float]], current_trend: DataTrend):
        """
        Prepares the feature matrix and target vector for least-squares optimization.
        """
        inputs = training_set.inputs
        outputs = training_set.outputs
        dqf = training_set.dqf
        
        valid_mask = (dqf == 1) if self.checkdqf else np.ones(len(dqf), dtype=bool)
            
        if not np.any(valid_mask):
            self.x_data, self.y_data = None, None
            return

        x_values = inputs[valid_mask]
        self.y_data = outputs[valid_mask]
        
        self.x_data, self.y_data = self.get_x_elements(x_values, self.y_data, current_trend)

    def residual(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the residual (error) between the model and actual data.
        Used by the least_squares optimizer.

        Args:
            params (np.ndarray): Candidate model coefficients.
            x (np.ndarray): Independent variable data.
            y (np.ndarray): Observed target values.

        Returns:
            np.ndarray: Vector of residuals.
        """
        return params[0] + params[1] * x - y

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Executes the polynomial fitting using robust non-linear least squares.

        Args:
            training_set (TrainingSet): Data points to train on.
            current_trend (DataTrend): Model object to populate.
            input_trend (Optional[DataTrend]): Baseline model for parameter 
                initialization.
        """
        try:
            logging.info(f"{CONTEXT}: Performing polynomial fit for {current_trend.mnemonic_id}")
            
            if self.params is not None:
                current_trend.set_params(self.params.tolist())
                logging.info(f"{CONTEXT}: Using pre-configured parameters.")
            else:
                init_params = input_trend.get_params() if input_trend else [1.0, 1.0]
                self._init_lsf_input(training_set, init_params, current_trend)
                
                if self.x_data is not None and self.x_data.shape[0] > 0:
                    # Extract the time-based independent variable
                    x_input = self.x_data[:, 1]
                    
                    # Optimization with 'soft_l1' loss for outlier robustness
                    res = least_squares(self.residual, init_params, args=(x_input, self.y_data), loss='soft_l1')
                    current_trend.set_params(res.x.tolist())
                    
                    logging.info(f"{CONTEXT}: Fit complete. Parameters: {res.x}")
                else:
                    logging.warning(f"{CONTEXT}: Insufficient data. Using mean baseline.")
                    params = np.zeros(self.m)
                    params[0] = current_trend.get_stat(MEAN)
                    current_trend.set_params(params.tolist())
                    
        except Exception as ex:
            logging.error(f"{CONTEXT}: Fitting failed: {ex}")
            raise Exception(f"Failed to perform polynomial trend training: {ex}")
