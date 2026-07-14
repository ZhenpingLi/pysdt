import logging
import os
import sys
import warnings
from typing import Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from config.sdt_constants import RETRAINSAMPLE, DEFAULTITER, MAXITER, NONLINEAR, INITTRAINSAMPLE
from sdtdb import sdt_db
from training.preprocessing import pre_processing
from training.preprocessing.orbit_based_transform import OrbitBasedTransform

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config.sdt_config as sdt_config
from algorithm.mnemonic_node import MnemonicNode
from algorithm.data_trend import DataTrend
from training.training_set import TrainingSet
from training.training_worker import TrainingWorker
from training.preprocessing.lttb_filter import LTTBFilter
from algorithm.fbnn.nn_init_training import NNInitTraining
from algorithm.fbnn.fbnn_trend import NeuralNet

# --- Constants ---
CONTEXT = "NNTrainingWorker"


class NNTrainingWorker(TrainingWorker):
    """
    Concrete TrainingWorker implementation for Neural Network (FBNN) models.
    
    This worker handles the training of multi-layer perceptron models for complex, 
    non-linear telemetry trends. It supports:
    1. **Initial Training**: Global optimization using multiple parallel random 
       restarts (via NNInitTraining).
    2. **Incremental Retraining**: High-precision tuning of existing weights 
       using the L-BFGS solver.
    3. **Orbit-Based Preprocessing**: Automatic normalization of time and values 
       within orbital cycles.
    """

    def __init__(self):
        """Initializes the NNTrainingWorker with its initialization sub-engine."""
        super().__init__()
        self.struct = None
        self.error_limit = 0.01
        self.params: Optional[np.ndarray] = None
        self.s_trend: Optional[LTTBFilter] = None
        self.downsample_scale = 1.0
        self.is_input_set = False
        self._init_training = NNInitTraining()
        thread_str = sdt_config.get_config_value("INITTRAINING")
        self.thread_training = thread_str == "THREAD"
        self.orbit_transform = None

    def set_config(self, node: MnemonicNode):
        """
        Configures the network topology and transformation parameters for the mnemonic.

        Args:
            node (MnemonicNode): The node to be trained.
        """
        super().set_config(node)
        self.num_training = 1
        
        # Get topology from database attributes (e.g., hidden layer sizes)
        nn_struct = self.algorithm.get_dimension()
        s_trend_period = sdt_config.get_float_property("TRAININGINTERNAL") / 60
        if s_trend_period > 0.0:
            self.s_trend = LTTBFilter(RETRAINSAMPLE)
            
        input_dim = 1 # Primary dimension is always model-time

        mn_type = sdt_db.get_mnemonic_type(node.name)
        if mn_type and mn_type.depends:
            input_dim += 1 # Add dependent telemetry as a second feature
                
        self.orbit_transform = OrbitBasedTransform(self.algorithm, node.name)
        self.struct = [input_dim, nn_struct[0], nn_struct[1], 1]
        
        self._init_training.set_config(self.algorithm, node.name)
        self._init_training.set_nn_struct(self.struct)

    def get_algorithm_type(self) -> int:
        """Returns the algorithm type code (NONLINEAR)."""
        return NONLINEAR

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Executes the neural network training pipeline.
        
        Orchestrates orbit transformation, data downsampling (using LTTB), 
        and either global initialization (if no baseline exists) or high-precision 
        retraining (if a baseline trend is provided).

        Args:
            training_set (TrainingSet): Source telemetry.
            current_trend (DataTrend): The model object to update.
            input_trend (Optional[DataTrend]): The baseline trend for retraining.
        """
        nn_trend = current_trend
        # 1. Apply orbital normalization
        self.orbit_transform.transform(training_set, self.get_pattern_offset_index(current_trend.get_reference_time()))
        
        # 2. Extract normalized features
        nn_set = pre_processing.get_nn_training_set(training_set, current_trend)
        if not nn_set:
            return
            
        nn_trend.set_scale_offset_list(nn_set.coef_list)
        
        init_sigma = 0.0
        init_params = input_trend.get_params() if input_trend is not None else None

        if init_params is not None:
            current_trend.set_params(init_params)
            init_sigma = self.calculate_sigma(training_set, current_trend)
            # if the init_sigma is the same as the statistical sigma
            # need to do the init training.
            sigma_t : float = input_trend.get_sigma_t()
            if sigma_t > 0 and float(init_sigma/sigma_t) > 0.9:
                init_params = None
                logging.info(f"{CONTEXT} the model standard deviation is close to the statistical standard deviation, start a new training")

        logging.info(f"{CONTEXT}: Starting training for {self.mnemonic_id} with {len(training_set.raw)} samples.")

        if init_params is not None:
            # --- Incremental Retraining ---
            # Downsample for faster L-BFGS convergence
            lttb_filter = LTTBFilter(n_out=RETRAINSAMPLE)
            sub_set = lttb_filter.filter(training_set=training_set)
            nn_set_retrain = pre_processing.get_nn_training_set(sub_set, current_trend)
            
            min_weight = self.retrain(nn_set_retrain, np.array(init_params))
            current_trend.set_params(min_weight)
            current_sigma = self.calculate_sigma(training_set, current_trend)
            
            # Revert if retraining significantly degraded the fit
            if current_sigma > init_sigma * 1.1:
                logging.info(f"{CONTEXT}: Retraining sigma ({current_sigma:.4f}) exceeded baseline. Reverting.")
                current_trend.set_params(init_params)
        else:
            # --- Global Initialization ---
            lttb_filter = LTTBFilter(n_out=INITTRAINSAMPLE)
            sub_set = lttb_filter.filter(training_set)
            nn_set_init = pre_processing.get_nn_training_set(sub_set, current_trend)
            self.init_training(nn_set_init, nn_trend, training_set)
            
        current_trend.set_trended(True)

    def retrain(self, training_set: TrainingSet, weights: np.ndarray) -> np.ndarray:
        """
        Performs high-precision tuning starting from a specific set of weights.

        Args:
            training_set (TrainingSet): Processed and downsampled dataset.
            weights (np.ndarray): The starting weight coefficients.

        Returns:
            np.ndarray: The optimized flat weight array.
        """
        net = NeuralNet(activation_func="TANH", structure=self.struct, max_iter=DEFAULTITER, tol=1.0e-5, warm_start=True, random_state=0)
        net.set_weight(weights)
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                net.model.fit(training_set.inputs, training_set.outputs)
        except Exception as e:
            logging.error(f"{CONTEXT}: Incremental fit failed: {e}")
        
        # Flatten and return the resulting scikit-learn parameters
        return np.concatenate([w.flatten() for w in net.model.coefs_] + [i.flatten() for i in net.model.intercepts_])

    def init_training(self, sub_set: TrainingSet, trend: DataTrend, full_set: TrainingSet):
        """
        Triggers the global search for an optimal starting point for new models.

        Args:
            sub_set (TrainingSet): Downsampled set for fast searching.
            trend (DataTrend): The model object to populate.
            full_set (TrainingSet): High-fidelity set for final refinement.
        """
        if self.thread_training:
            # Use multi-threaded random restart search
            nn_set_full = pre_processing.get_nn_training_set(full_set, trend)
            min_weight = self._init_training.perform_init_training(sub_set, nn_set_full)
        else:
            # Serial fallback
            min_weight = self._training(sub_set)
            # Refine on full data
            net = NeuralNet("TANH", self.struct, MAXITER, 1.0e-6, warm_start=True)
            net.set_weight(min_weight)
            net.model.fit(full_set.inputs, full_set.outputs)
            min_weight = np.concatenate([w.flatten() for w in net.model.coefs_] + [i.flatten() for i in net.model.intercepts_])

        trend.set_params(min_weight)

    def _training(self, training_set: TrainingSet) -> np.ndarray:
        """Serial random-restart search fallback."""
        min_error = float('inf')
        min_weight = None
        
        net = NeuralNet("TANH", self.struct, MAXITER, 1.0e-5, False, 0)
        for _ in range(100): # Hardcoded limit for serial fallback
            net.model.fit(training_set.inputs, training_set.outputs)
            if net.model.loss_ < min_error:
                min_error = net.model.loss_
                min_weight = np.concatenate([w.flatten() for w in net.model.coefs_] + [i.flatten() for i in net.model.intercepts_])
        return min_weight

    def get_pattern_offset_index(self, input_ref_time: float) -> int:
        """
        Calculates the index of the pattern cycle containing the reference time.
        """
        if self.pattern_times is None:
            return 0
            
        diffs = np.abs(self.pattern_times - input_ref_time)
        index = np.argmin(diffs)
        
        if index > 0:
            index += sdt_config.get_int_property("NUMPATTERNINTRAINING") or 1
            
        return int(index)
