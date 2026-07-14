import logging
import os
import random
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np

from config.sdt_constants import MAXITER

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from algorithm.algorithm_def import AlgorithmDef
from training.training_set import TrainingSet
from algorithm.fbnn.neural_net_wrapper import NeuralNet

# --- Constants ---
CONTEXT = "NNInitTraining"


class NNInitTraining:
    """
    Orchestrator for the global initialization of Neural Network models.
    
    To overcome the local minima problem in gradient descent, this class 
    executes multiple training sessions ('num_iterations') in parallel with 
    different random weight initializations. It identifies the best performing 
    weights on a downsampled dataset and then performs a final high-precision 
    refinement on the full training set.
    """

    def __init__(self):
        """Initializes the initialization engine with default parameters."""
        self.num_iterations = 50
        """Total number of random restarts."""
        self.num_threads = 10
        """Maximum parallel threads for initial fitting."""
        self.error_limit = 1e-5
        """Early stopping threshold for the initial phase."""
        self.nn_struct: Optional[List[int]] = None
        self.algorithm: Optional[AlgorithmDef] = None
        self.mnemonic_id: Optional[str] = None

    def set_config(self, alg: AlgorithmDef, mnemonic_id: str):
        """
        Configures the engine with algorithm and mnemonic metadata.

        Args:
            alg (AlgorithmDef): The algorithm definition.
            mnemonic_id (str): The identifier for the mnemonic.
        """
        self.algorithm = alg
        self.mnemonic_id = mnemonic_id

    def set_nn_struct(self, nn_struct: List[int]):
        """
        Defines the network topology to be initialized.

        Args:
            nn_struct (List[int]): Topology, e.g., [1, 5, 5, 1].
        """
        self.nn_struct = nn_struct

    def set_error_limit(self, limit: float):
        """
        Sets the loss threshold for early exit.

        Args:
            limit (float): Targeted minimum loss value.
        """
        self.error_limit = limit

    def perform_init_training(self, training_set: TrainingSet, refined_set: TrainingSet) -> Optional[np.ndarray]:
        """
        Executes the multi-threaded global search for optimal starting weights.
        
        This method manages the thread pool, collects results, identifies the 
        global minimum found across all iterations, and triggers the final 
        refinement on the full dataset.

        Args:
            training_set (TrainingSet): Typically a downsampled dataset for fast iteration.
            refined_set (TrainingSet): The full, original dataset for final tuning.

        Returns:
            Optional[np.ndarray]: The best flat weight array found, or None if failed.
        """
        min_loss = float('inf')
        min_weight = None

        logging.info(f"{CONTEXT}: Starting {self.num_iterations} parallel iterations for {self.mnemonic_id}.")

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Step 1: Broad search using parallel threads
            futures = {
                executor.submit(self._train_single_network, training_set, random.randint(0, 100000)): i 
                for i in range(self.num_iterations)
            }
            
            for future in as_completed(futures):
                try:
                    loss, weight = future.result()
                    if loss < min_loss:
                        min_loss = loss
                        min_weight = weight
                        logging.info(f"{CONTEXT}: Iteration {futures[future]}: New global minimum found ({min_loss:.6f}).")
                        
                        # Early exit if we find a sufficiently good fit
                        if min_loss < self.error_limit:
                            logging.info(f"{CONTEXT}: Error limit reached. Terminating remaining iterations.")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                            
                except Exception as e:
                    logging.error(f"{CONTEXT}: A training thread failed: {e}")

        # Step 2: Final refinement on the high-fidelity dataset
        if min_weight is not None:
            logging.info(f"{CONTEXT}: Refining best initialization with full dataset.")
            net = NeuralNet("TANH", self.nn_struct, max_iter=MAXITER, tol=1e-6, warm_start=True)
            net.set_weight(min_weight)
            
            try:
                net.fit(refined_set.inputs, refined_set.outputs)
                final_loss = net.model.loss_
                # Flatten the internal scikit-learn coefficients and intercepts into a 1D array
                final_weights = np.concatenate([w.flatten() for w in net.model.coefs_] + 
                                             [i.flatten() for i in net.model.intercepts_])
                logging.info(f"{CONTEXT}: Refinement complete. Final loss: {final_loss:.6f}")
                return final_weights
            except Exception as e:
                logging.error(f"{CONTEXT}: Final refinement failed ({e}). Returning unrefined best weight.")
                return min_weight
            
        return None

    def _train_single_network(self, training_set: TrainingSet, random_seed: int) -> Tuple[float, np.ndarray]:
        """
        Trains a single candidate network instance.

        Args:
            training_set (TrainingSet): Dataset for fitting.
            random_seed (int): Seed for the weight initialization.

        Returns:
            Tuple[float, np.ndarray]: A tuple of (loss_value, weights_array).
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        net = NeuralNet("TANH", self.nn_struct, max_iter=MAXITER, tol=1e-6, warm_start=False, random_state=random_seed)
        net.fit(training_set.inputs, training_set.outputs)
        
        loss = net.model.loss_
        weights = np.concatenate([w.flatten() for w in net.model.coefs_] + 
                                 [i.flatten() for i in net.model.intercepts_])
        
        return loss, weights
