from typing import List, Optional
import sys
import os
import logging
import numpy as np

from algorithm.mnemonic_node import MnemonicNode

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.training_worker import TrainingWorker
from algorithm.data_trend import DataTrend
from training.training_set import TrainingSet
from algorithm.lnn.lnn_trend import LNNTrend
from algorithm.lnn.lnn_trainer import LNNTrainer

# Constants
CONTEXT = "LNNTrainingWorker"
NONLINEAR = 2 # Assuming a value

class LNNTrainingWorker(TrainingWorker):
    """
    A specialized implementation of the TrainingWorker for Liquid Neural Network (LNN) models.
    """

    def __init__(self):
        super().__init__()
        self.epochs = 100 # Default number of training epochs
        self.learning_rate = 0.01
        self.decay_rate = 0.001
        self.momentum = 0.9

    def set_config(self, node: MnemonicNode):
        super().set_config(node)
        # You can add LNN-specific configurations here from the algorithm attributes
        self.epochs = int(self.algorithm.get_attribute("epochs") or 100)
        self.learning_rate = float(self.algorithm.get_attribute("learning_rate") or 0.01)
        self.decay_rate = float(self.algorithm.get_attribute("decay_rate") or 0.001)
        self.momentum = float(self.algorithm.get_attribute("momentum") or 0.9)

    def get_algorithm_type(self) -> int:
        return NONLINEAR

    def do_data_training(self, training_set: TrainingSet, current_trend: DataTrend, input_trend: Optional[DataTrend]):
        """
        Performs the main data training process for the LNN model.
        """
        if not isinstance(current_trend, LNNTrend):
            raise Exception("LNNTrainingWorker requires an LNNTrend instance.")
        
        lnn_trend = current_trend
        lnn_model = lnn_trend.lnn
        
        if lnn_model is None:
            raise Exception("LNN model is not initialized in the LNNTrend object.")

        # Initialize the trainer
        trainer = LNNTrainer(
            lnn=lnn_model,
            initial_learning_rate=self.learning_rate,
            decay_rate=self.decay_rate,
            momentum=self.momentum
        )

        # Prepare the data for sequence-based training
        # The 'inputs' from TrainingSet are the features (e.g., normalized time)
        # The 'outputs' are the target values
        inputs = training_set.get_inputs()
        targets = training_set.get_outputs()
        
        if inputs is None or targets is None:
            logging.warning(f"{CONTEXT}: Training set has no data.")
            return
            
        # The LNN expects a sequence of 1D arrays for input.
        # Our training set inputs are likely (N, 1) for time, so we can iterate through them.
        input_sequence = [row for row in inputs]
        target_sequence = [np.array([val]) for val in targets] # Ensure targets are also 1D arrays

        logging.info(f"{CONTEXT}: Starting LNN training for {self.epochs} epochs...")

        # --- Training Loop ---
        for epoch in range(self.epochs):
            # The trainer's `train_sequence` method handles one full pass (forward and backward)
            trainer.train_sequence(input_sequence, target_sequence)
            
            if (epoch + 1) % 10 == 0:
                # Optional: Calculate and log loss
                # This would require another forward pass without training to get current loss
                logging.info(f"{CONTEXT}: Epoch {epoch + 1}/{self.epochs} completed.")

        logging.info(f"{CONTEXT}: LNN training finished.")

        # After training, the LNN model inside the trainer has the updated weights.
        # We now update the trend object with these trained parameters.
        final_weights = lnn_model.get_weights_flat()
        lnn_trend.set_params(final_weights.tolist())
        lnn_trend.set_trended(True)
