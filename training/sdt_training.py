import argparse
import logging
import os
import sys
import traceback
from typing import Dict, Optional, List

import numpy as np

import orbit.orbit_model_manager
from algorithm.algorithm_def import AlgorithmDef
from algorithm.data_trend import DataTrend, DEFAULT
from config.sdt_constants import DAY_IN_SECONDS
from dataio.data_training_io import DataTrainingIO
from training.training_set import TrainingSet

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.sdt_config as sdt_config
from training import data_buffer as db
import plugin_manager
from algorithm.mnemonic_node import MnemonicNode
from algorithm.trend_tree import TrendTree
from training.training_worker import TrainingWorker
from sdt_exception import SDTException

# Constants
CONTEXT = "SDTTraining"
LONGTERM = 1
TRAININGPARAM = "trainingparam"

class SDTTraining:
    """
    Main orchestrator for the telemetry data training process.
    
    This class coordinates the execution of training algorithms for individual 
    mnemonics or subsystems. It handles the retrieval of baseline (input) models, 
    manages worker instances, and calculates historical statistical baselines 
    (sigma means) required for health evaluation.
    """

    def __init__(self, input_time: float, input_id: str, trend_io: DataTrainingIO) -> None:
        """
        Initializes the SDTTraining orchestrator.

        Args:
            input_time (float): Reference timestamp for the input baseline model.
            input_id (str): ID of the input model to use as a starting point.
            trend_io (DataTrainingIO): I/O manager for retrieving/saving trends.
        """
        self.training_map: Dict[str, TrainingWorker] = {}
        self.input_time = input_time
        self.input_id = input_id
        self.trend_input_io = trend_io
        self.root: Optional[MnemonicNode] = None
        
        retrain_threshold_str = sdt_config.get_config_value("SIGMARATIOTHRESHOLD")
        self.retrain_threshold = float(retrain_threshold_str) if retrain_threshold_str else 2.0

    def train(self, node: MnemonicNode, algorithm: AlgorithmDef, training_set: TrainingSet):
        """
        Executes the training pipeline for a specific mnemonic node.
        
        It retrieves the baseline trend, calculates historical sigma means, 
        identifies the appropriate training worker, and triggers the actual 
        model fitting.

        Args:
            node (MnemonicNode): The tree node to be trained.
            algorithm (AlgorithmDef): The algorithm configuration for this mnemonic.
            training_set (TrainingSet): The session telemetry data.

        Raises:
            SDTException: If no worker is found or if a critical training error occurs.
        """
        if not node.is_training_defined():
            logging.warning(f"{CONTEXT}: Training is not defined for {node.name}")
            return None

        self.root = node
        if db.session_type == LONGTERM:
            # For long-term sessions, ensure global time reflects the data range
            time_end = training_set.inputs[-1, 0] if training_set.inputs.size > 0 else 0.0
            db.set_session_time(time_end)
            
        try:
            # 1. Retrieve baseline trend for incremental training
            input_trend = self._get_input_trend(algorithm)
            if input_trend:
                node.set_input(input_trend)
                
            # 2. Calculate historical standard deviation for status evaluation
            sigma_mean = self.get_sigma_mean(node.name, algorithm)
            node.sigma_mean = sigma_mean
            
            if not algorithm:
                logging.info(f"{CONTEXT}: Algorithm definition missing for {node.name}")
                return None

            # 3. Fetch worker and execute training
            worker = self._get_training_worker(algorithm.get_name())
            if not worker:
                raise SDTException(f"Failed to load training worker for algorithm: {algorithm.get_name()}", CONTEXT)

            worker.do_training(training_set, node)
            
        except SDTException as e:
            logging.error(f"{CONTEXT}: SDT Error for {node.name}: {e}")
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"{CONTEXT}: Unexpected training error for {node.name}: {e}\n{error_details}")
            raise SDTException(f"Unexpected error during training: {e}", CONTEXT)

    def _get_input_trend(self, algorithm: AlgorithmDef) -> Optional[DataTrend]:
        """
        Retrieves the baseline model that serves as the starting point for training.
        
        It searches the in-memory buffer or queries the data archive based on 
        the session's temporal context and the algorithm's requirements.

        Args:
            algorithm (AlgorithmDef): The current algorithm configuration.

        Returns:
            Optional[DataTrend]: The reconstructed baseline model, or None if not found.
        """
        in_time = self.input_time
        _num_pattern_in_training = sdt_config.num_pattern_in_training
        if algorithm and algorithm.np > 0:
            _num_pattern_in_training = algorithm.np
            
        model_time = db.get_default_model_time(algorithm)
        pattern_period = model_time.get_model_period()
        
        if in_time == 0:
            in_time = db.session_time
            
        is_orbital_based = algorithm.get_name() == "fbnn"
        input_times = orbit.orbit_model_manager.get_input_trend_times(
            pattern_period=pattern_period, 
            session_time=in_time, 
            num_pattern_in_training=_num_pattern_in_training, 
            is_orbitbased=is_orbital_based
        )
        
        if input_times.any():
            if self.input_id:
                # Check for manually specified input model ID
                data_trend = db.get_input_trend(self.input_id)
                if data_trend:
                    return data_trend
                else:
                    algorithm_data = self.trend_input_io.get_data_trend(mnemonic_id=self.root.name, input_times=input_times.tolist(), state=DEFAULT)
                    if algorithm_data:
                        data_trend = plugin_manager.get_data_trend(algorithm_data.alg_name, algorithm_data.mnemonic_id)
                        data_trend.set_algorithm_data(algorithm_data)
                    return data_trend
            else:
                # Default logic: get the most recent model from the archive
                if not db.is_manual_training:
                    algorithm_data = self.trend_input_io.get_input_data_trend(self.root.name, input_times.tolist())
                    if algorithm_data:
                        try:
                            data_trend = plugin_manager.get_data_trend(algorithm_data.alg_name, algorithm_data.mnemonic_id)
                            data_trend.set_algorithm_data(algorithm_data)
                            return data_trend
                        except Exception as e:
                            logging.error(f"{CONTEXT}: Reconstructing baseline failed for {self.root.name}: {e}")

        logging.info(f"{CONTEXT}: No baseline input trend found for {self.root.name}")
        return None

    def _get_training_worker(self, alg_name: str) -> Optional[TrainingWorker]:
        """
        Retrieves a cached worker instance for the specified algorithm name.

        Args:
            alg_name (str): The name of the algorithm (e.g., 'fbnn').

        Returns:
            Optional[TrainingWorker]: The worker instance.
        """
        if alg_name not in self.training_map:
            training_worker = plugin_manager.get_training_worker(alg_name)
            self.training_map[alg_name] = training_worker
        return self.training_map[alg_name]

    def get_sigma_mean(self, mnemonic_id: str, algorithm: AlgorithmDef) -> Dict[str, float]:
        """
        Calculates historical average standard deviations for all states of a mnemonic.

        Args:
            mnemonic_id (str): ID of the mnemonic.
            algorithm (AlgorithmDef): Algorithm definition.

        Returns:
            Dict[str, float]: Mapping of state names to their historical sigma means.
        """
        sigma_mean_dict: Dict[str, float] = {}
        state_array = algorithm.get_states()
        if state_array:
            for state in state_array:
                sigma_mean_dict[state.name] = self.get_sigma_mean_for_state(mnemonic_id, state.name)
        else:
            sigma_mean_dict[DEFAULT] = self.get_sigma_mean_for_state(mnemonic_id, DEFAULT)
        return sigma_mean_dict

    def get_sigma_mean_for_state(self, mnemonic_id: str, w_state: str) -> float:
        """
        Retrieves standard deviation history from the archive and calculates the mean.

        Args:
            mnemonic_id (str): ID of the mnemonic.
            w_state (str): Operational state name.

        Returns:
            float: The calculated sigma mean, or 0.0 if insufficient history.
        """
        # Look back 180 days for historical baseline
        start = db.session_time - 180 * DAY_IN_SECONDS
        sigma_list = self.trend_input_io.get_sigma(mnemonic_id, start, db.session_time, w_state)
        
        if sigma_list:
            sigma_values = np.array([dp.data[0] for dp in sigma_list if dp.data[0] > 0])
            if sigma_values.size > 0:
                mean = np.mean(sigma_values)
                last_sigma = sigma_list[-1].data[0]
                # Return the larger of the two to be conservative in monitoring
                return float(max(mean, last_sigma))
        return 0.0
    
def main():
    """
    Main entry point for standalone training execution (testing).
    """
    parser = argparse.ArgumentParser(description="SDT Training Manager")
    parser.add_argument("group", help="The subsystem or mnemonic ID to train")
    parser.add_argument("-t", "--type", type=int, default=0, help="0: ShortTerm, 1: LongTerm")
    parser.add_argument("-s", "--satid", help="Satellite ID override")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    sdt_config.set_sat_id(args.satid or args.group)
    db.set_session_type(args.type)
    db.set_session_time(0)

    # Note: Full execution requires a configured DataTrainingIO instance.
    logging.info("Standalone training run initiated.")

if __name__ == "__main__":
    main()
