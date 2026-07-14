import logging
import numpy as np
from typing import List, Optional
import sys
import os

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.training_worker import TrainingWorker
from training.training_set import TrainingSet
from algorithm.data_trend import DataTrend
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_point import DataPoint
from training import data_buffer
from config.sdt_constants import LINEAR, MEAN

# --- Constants ---
CONTEXT = "TimePolyTrendWorker"


class TimePolyTrendWorker(TrainingWorker):
    """
    Concrete TrainingWorker implementation for Time-based Polynomial trends.
    
    This worker fits a polynomial function where the independent variable is 
    absolute time. It constructs a Vandermonde-like design matrix and uses 
    NumPy's optimized linear least squares solver (lstsq) to determine the 
    coefficients.
    """

    def __init__(self):
        """Initializes the TimePolyTrendWorker."""
        super().__init__()
        self.m: int = 0
        self.s_index: int = 0
        self.ref_time: float = 0.0

    def set_config(self, node: MnemonicNode):
        """
        Configures the worker with parameters from the algorithm definition.
        
        Determines the polynomial order and starting index from the 
        algorithm attributes and initializes the temporal context.

        Args:
            node (MnemonicNode): The mnemonic node to be trained.
        """
        super().set_config(node)
        
        # dimensionality: [order] or [s_index, order]
        dim = self.algorithm.get_dimension()
        if len(dim) == 1:
            self.s_index = 0
            self.m = dim[0]
        elif len(dim) >= 2:
            self.s_index = dim[0]
            self.m = dim[1]
            
        self.ref_time = data_buffer.session_start
        self.num_training = 1
        self.num_pattern_in_training = 1
        
        # Simple session-wide boundaries for time-based fitting
        self.pattern_times = np.array([data_buffer.session_start, data_buffer.session_end])

    def get_algorithm_type(self) -> int:
        """Returns the algorithm type code (LINEAR)."""
        return LINEAR

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend] = None):
        """
        Executes the polynomial fitting using linear least squares.
        
        This method:
        1. Normalizes input time relative to the model's reference time.
        2. Constructs the design matrix A containing expansion terms (t^s, t^s+1...).
        3. Solves the system: y = Ap + ε for coefficients 'p'.
        4. Populates the model object with the resulting parameters.

        Args:
            training_set (TrainingSet): Data points to train on.
            current_trend (DataTrend): Model object to populate.
            input_trend (Optional[DataTrend]): Unused baseline model.
        """
        logging.info(f"[{current_trend.mnemonic_id}] {CONTEXT}: Performing time-based polynomial fit.")
        
        self.ref_time = current_trend.get_reference_time()
        
        inputs = training_set.inputs
        outputs = training_set.outputs
        
        if inputs is None or outputs is None or inputs.size == 0:
            logging.warning(f"{CONTEXT}: Empty training set for {current_trend.mnemonic_id}")
            return

        # Prepare independent variable (relative time) and target values
        # Assume first column of inputs is time
        x_data = inputs[:, 0] - self.ref_time
        y_data = outputs

        try:
            # 1. Generate array of powers: [s_index, s_index+1, ..., s_index+m-1]
            powers = np.arange(self.s_index, self.s_index + self.m)
            
            # 2. Construct the design matrix A (shape: samples x m)
            # A_ij = x_i ^ power_j
            A = np.power(x_data[:, None], powers)
            
            # 3. Solve the system using singular value decomposition (SVD)
            params, residuals, rank, _ = np.linalg.lstsq(A, y_data, rcond=None)
            
            # Update the trend model
            current_trend.set_params(params.tolist())
            
            logging.debug(f"{CONTEXT}: Optimization complete. Rank: {rank}, Residuals: {residuals}")
            for i, p in enumerate(params):
                logging.info(f"{CONTEXT}: Coefficient w_{i} = {p:.4e}")

        except np.linalg.LinAlgError as e:
            logging.error(f"{CONTEXT}: Linear algebra failure: {e}")
            # Robust fallback: set constant intercept to mean if possible
            params = [0.0] * self.m
            if self.s_index == 0:
                params[0] = current_trend.get_stat(MEAN)
            current_trend.set_params(params)
        except Exception as e:
            logging.error(f"{CONTEXT}: Unexpected fitting error: {e}", exc_info=True)
