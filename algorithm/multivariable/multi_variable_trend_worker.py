import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from algorithm.algorithm_def import AlgorithmDef
from algorithm.multivariable.multi_variable_trend import MultiVariableTrend
from sdtdb import sdt_db

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.ridgeregression.ridge_reg_training_worker import RidgeRegTrainingWorker
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_trend import DataTrend
from training.training_set import TrainingSet
import training.data_buffer


# --- Constants ---
CONTEXT = "MultiVariableTrendWorker"
LINEAR = 1
MEAN = "mean"


def dynamic_multi_variable_fit(x: np.ndarray, *params) -> np.ndarray:
    """
    Generalized multi-variable linear function for curve fitting.
    
    Formula: y = w0 + w1*x1 + w2*x2 + ... + wm*xm

    Args:
        x (np.ndarray): Transposed input matrix with m rows (variables) and N columns.
        *params: Variable length coefficients [w0, w1, ..., wm].

    Returns:
        np.ndarray: Predicted values of length N.
    """
    # params[0] is the intercept (bias)
    result = params[0]
    # Iterate through each independent variable
    for i in range(1, len(params)):
        # params[i] is the weight for the variable at x[i-1]
        result += params[i] * x[i-1]
    return result


class MultiVariableTrendWorker(RidgeRegTrainingWorker):
    """
    TrainingWorker implementation for Multi-Variable Polynomial Trends.
    
    This worker optimizes coefficients for a linear model with multiple 
    independent variables using scipy.optimize.curve_fit. It dynamically 
    adapts to the number of variables specified in the algorithm definition.
    """

    def __init__(self):
        """Initializes the MultiVariableTrendWorker."""
        super().__init__()
        self.m: int = 0
        self.s_index: int = 0
        self.checkdqf: bool = False
        self.params: Optional[np.ndarray] = None

    def set_config(self, node: MnemonicNode):
        """Configures the worker for multi-variable modeling."""
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

        # Determine dimensionality m (number of variables)
        dim = self.algorithm.get_dimension()
        if len(dim) == 1:
            self.s_index = 0
            self.m = dim[0]
        elif len(dim) >= 2:
            self.s_index = dim[0]
            self.m = dim[1]
        num_depends = 1
        depends = sdt_db.get_mnemonic_type(self.mnemonic_id).depends
        if depends:
            num_depends = len(depends.split('|'))
        self.param_size = self.m * num_depends + 1

        self.checkdqf = self.algorithm.check_dqf()
        
        # Default 48h pattern window
        self.pattern_times = np.array([
            training.data_buffer.session_end - 2 * self.num_pattern_in_training * self.pattern_period,
            training.data_buffer.session_end - self.num_pattern_in_training * self.pattern_period,
            training.data_buffer.session_end
        ])

    def get_algorithm_type(self) -> int:
        """Returns the algorithm type code (LINEAR)."""
        return LINEAR

    def get_x_elements(self, inputs: np.ndarray, outputs: np.ndarray, current_trend: DataTrend) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms raw inputs into the polynomial feature space."""
        poly_trend: MultiVariableTrend = current_trend
        # Returns [1.0, v1, v2, ...] for each row
        x_elements = np.array([poly_trend.poly_function(i) for i in inputs])
        return x_elements, outputs

    def _init_lsf_input(self, training_set: TrainingSet, init_params: Optional[List[float]], current_trend: DataTrend):
        """Prepares the feature matrix and target vector for optimization."""
        inputs = training_set.inputs
        outputs = training_set.outputs
        dqf = training_set.dqf
        
        valid_mask = (dqf == 1) if self.checkdqf else np.ones(len(dqf), dtype=bool)
            
        if not np.any(valid_mask):
            self.x_data, self.y_data = None, None
            return

        x_values = inputs[valid_mask]
        y_values = outputs[valid_mask]
        
        self.x_data, self.y_data = self.get_x_elements(x_values, y_values, current_trend)

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """Executes the multi-variable optimization using curve_fit."""
        try:
            logging.info(f"{CONTEXT}: Starting multi-variable fit for {current_trend.mnemonic_id} (m={self.m})")

            if self.params is not None:
                current_trend.set_params(self.params.tolist())
            else:
                init_params = input_trend.get_params() if input_trend else None
                self._init_lsf_input(training_set, init_params, current_trend)

                # Capture x_data in a local variable to satisfy strict type linters
                x_data_local = self.x_data

                if x_data_local is not None and x_data_local.shape[0] > 0:
                    # x_data contains [1.0, v1, v2, ...].
                    # curve_fit function needs x inputs to be (m, N)
                    # noinspection PyUnresolvedReferences
                    x_input = x_data_local[:, 1:].T

                    # Initial guess: intercept=mean, others=1.0
                    if init_params is not None and len(init_params) == self.param_size:
                        i_params = init_params
                    else:
                        i_params = [float(current_trend.get_stat(MEAN))] + [1.0] * (self.param_size-1)

                    # Perform optimization.
                    # Fixed Unpacking warning by grabbing the sequence result first
                    fit_result = curve_fit(dynamic_multi_variable_fit, x_input, self.y_data, p0=i_params)
                    popt = fit_result[0]

                    current_trend.set_params(popt.tolist())
                    logging.info(f"{CONTEXT}: Optimization complete. Coefficients: {popt}")
                else:
                    logging.warning(f"{CONTEXT}: No valid data. Using mean baseline.")
                    params = np.zeros(self.m + 1)
                    params[0] = current_trend.get_stat(MEAN)
                    current_trend.set_params(params.tolist())
        except Exception as ex:
            logging.error(f"{CONTEXT}: Training failed for {current_trend.mnemonic_id}: {ex}")
            raise Exception(f"Failed multi-variable fit: {ex}")

