import argparse
import logging
import os
import sys
import traceback
import warnings
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import ray

from algorithm.subsystem_output import SubsystemOutput
from algorithm.training_output import TrainingOutputData
from algorithm.trend_node import TrendNode
from config.sdt_constants import HOUR_IN_SECONDS
from posttraining.post_training_processing_manager import post_training_analysis
from training.data_training_thread import run_training_task

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.sdt_config as sdt_config
from sdtdb import sdt_db
from training import data_buffer
from util import time_util

# --- Module-level Session State ---
CONTEXT = "SDTTrainingSession"

is_complete: bool = False
"""Flag indicating if the current session has finished processing all tasks."""


@ray.remote
def run_training_task_remote(item_id: str, input_time: float, input_id: str, current_sat_id: str, daily_time: float, is_manual_training: bool) -> Tuple[str, List[TrainingOutputData]]:
    """
    Ray remote wrapper for the run_training_task function.
    
    This function is executed in parallel by Ray workers across multiple CPU cores.

    Args:
        item_id (str): The ID of the subsystem or mnemonic to train.
        input_time (float): Reference time for the input model.
        input_id (str): ID of the input trend model.
        current_sat_id (str): The active satellite identifier.
        daily_time (float): The current session's reference time.
        is_manual_training (bool): Flag indicating manual trigger mode.

    Returns:
        Tuple[str, List[TrainingOutputData]]: A tuple of (item_id, training_results).
    """
    logger = logging.getLogger(__name__)

    # 2. CRITICAL: If the worker reset the level, force it back to INFO
    if not logger.handlers and len(logging.getLogger().handlers) == 0:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logger.setLevel(logging.INFO)
    return item_id, run_training_task(item_id, input_time, input_id, current_sat_id, daily_time, is_manual_training)

def create_trend_node(trained_output_list: List[TrainingOutputData]) -> TrendNode:
    """
    Constructs a hierarchical TrendNode from a list of mnemonic training results.

    Args:
        trained_output_list (List[TrainingOutputData]): Results for individual mnemonics.

    Returns:
        TrendNode: The root node of the reconstructed subsystem hierarchy.
    """
    trend_node : TrendNode = None
    from algorithm.mnemonic_node import MnemonicNode
    for output in trained_output_list:
        if trend_node is None:
            mnemonic_id : str = output.mnemonic_id
            subsystem_name = sdt_db.get_subsystem_for_mnemonic(mnemonic_id)
            trend_node = TrendNode(subsystem_name)
        mnemonic_node = MnemonicNode(output.mnemonic_id)
        trend_node.add_child(mnemonic_node)
        mnemonic_node.set_training_output(output)
    return trend_node


def create_subsystem_output(item_id: str, training_output_list: List[TrainingOutputData]) -> SubsystemOutput:
    """
    Factory function to create a SubsystemOutput object.

    Args:
        item_id (str): The identifier (subsystem or mnemonic).
        training_output_list (List[TrainingOutputData]): The training results.

    Returns:
        SubsystemOutput: The aggregated output container.
    """
    if sdt_db.is_subsystem(item_id):
        subsystem_id = item_id
    else:
        subsystem_id = sdt_db.get_subsystem_for_mnemonic(item_id)
    return SubsystemOutput(subsystem_name=subsystem_id, mnemonic_output_list=training_output_list)


class SDTTrainingSession:
    """
    Orchestrator for telemetry data training sessions.
    
    This class manages the full lifecycle of a training session:
    1. Parsing command-line arguments and configuration.
    2. Initializing the parallel execution environment (Ray).
    3. Iterating through time ranges and dispatching parallel tasks to workers.
    4. Aggregating results and triggering post-training analysis.
    """

    def __init__(self):
        """
        Initializes the SDTTrainingSession and the Ray cluster.
        """
        self.id_list: List[str] = []
        ray.init(local_mode=False)
        # Determine number of workers based on CPU cores or config
        self.max_workers = sdt_config.get_int_property("MAXWORKERS") or os.cpu_count() or 4

    def perform_training(self, args: argparse.Namespace):
        """
        Entry point to start the training session based on provided arguments.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        self._parse_args(args)
        
        start_time = args.start or 0.0
        end_time = args.end or 0.0
        input_time = args.input_time or 0.0
        
        if start_time == 0 and end_time > 0:
            start_time = end_time
        elif start_time > 0 and end_time == 0:
            end_time = start_time
        elif start_time == 0 and end_time == 0:
            now = datetime.now(timezone.utc).timestamp()
            end_time = start_time = time_util.get_day_start(now)

        if not self.id_list:
            self._get_all_group_list()
            
        self._process_training_range(args.inputid, start_time, end_time, input_time)

    def _process_training_range(self, input_id: Optional[str], start: float, end: float, input_time: float):
        """
        Iterates through the training time range and launches parallel tasks for each delta.

        Args:
            input_id (Optional[str]): ID of the baseline model.
            start (float): Session start timestamp.
            end (float): Session end timestamp.
            input_time (float): Reference time for baseline models.
        """
        global is_complete
        is_complete = False
        
        logging.info(f"{CONTEXT}: Training Start: {time_util.get_time_tag_from_seconds(start)} End: {time_util.get_time_tag_from_seconds(end)}")

        if len(self.id_list) > sdt_config.BATCHNUM:
            self._sort_dataset()
            
        training_delta = data_buffer.get_training_delta() * HOUR_IN_SECONDS
        daily_time = start
        current_sat_id = sdt_config.sat_id

        while daily_time <= end:
            if not data_buffer.is_manual_training:
                data_buffer.clear_buffer()
            
            data_buffer.set_session_time(daily_time)
            session_string = time_util.get_simple_time_tag_from_seconds(data_buffer.session_time)
            logging.info(f"{CONTEXT}: Starting session for {session_string}")
            
            num_tasks = len(self.id_list)
            current_workers = min(num_tasks, self.max_workers)

            logging.info(f"{CONTEXT}: Dispatching {num_tasks} tasks to {current_workers} workers.")

            # Create Ray remote futures
            future_refs = [
                run_training_task_remote.remote(item_id, input_time, input_id, current_sat_id, daily_time, data_buffer.is_manual_training)
                for item_id in self.id_list
            ]

            subsystem_output_list : List[SubsystemOutput] = []

            # Gather results as they complete
            for result_tuple in ray.util.as_completed(future_refs):
                try:
                    item_id, trained_output_list = result_tuple
                    if trained_output_list:
                        subsystem_output = create_subsystem_output(item_id, trained_output_list)
                        data_buffer.add_subsystem_output(subsystem_output)
                        subsystem_output_list.append(subsystem_output)
                    else:
                        logging.warning(f"{CONTEXT}: Task for {item_id} returned no results.")

                except Exception:
                    error_stack = traceback.format_exc()
                    logging.error(f"{CONTEXT}: Critical worker failure: {error_stack}")

            # Trigger post-training analysis for the session results
            if subsystem_output_list:
                post_training_analysis(subsystem_output_list)
            
            daily_time += training_delta

        is_complete = True
        ray.shutdown()

    def _parse_args(self, args: argparse.Namespace):
        """
        Internal helper to populate the training list from command-line arguments.
        """
        data_buffer.is_manual_training = (args.mode == 'manual')

        if args.ids:
            for id_str in args.ids:
                if sdt_db.exist(id_str):
                    self.id_list.append(id_str)
                else:
                    logging.warning(f"{CONTEXT}: ID '{id_str}' not found in database.")

    def _get_all_group_list(self):
        """Loads all available subsystem IDs from the database."""
        self.id_list = [name for name in sdt_db.get_dataset_name_list()]

    def _sort_dataset(self):
        """
        Sorts subsystems by the number of mnemonics (descending).
        Used to optimize worker utilization by starting largest tasks first.
        """
        self.id_list.sort(key=lambda lg: len(sdt_db.get_dataset(lg).mnemonics), reverse=True)

    def close(self):
        """Release session resources."""
        pass


def main():
    """
    Main entry point for standalone session execution.
    """
    parser = argparse.ArgumentParser(description="SDT Training Session Manager")
    parser.add_argument("ids", nargs='*', help="List of IDs to train")
    parser.add_argument("-t", "--type", type=int, default=0, choices=[0, 1])
    parser.add_argument("-s", "--start", type=str)
    parser.add_argument("-e", "--end", type=str)
    parser.add_argument("--input_time", type=str)
    parser.add_argument("--inputid", help="Input model ID")
    parser.add_argument("--mode", choices=['auto', 'manual'], default='auto')
    parser.add_argument("--satid", default="G16")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    sdt_config.set_sat_id(args.satid)
    data_buffer.set_session_type(args.type)

    session = SDTTrainingSession()
    try:
        session.perform_training(args)
    except Exception as e:
        logging.error(f"{CONTEXT}: Session failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
