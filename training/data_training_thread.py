import argparse
import logging
import os
import re
import sys
import traceback
from typing import List, Optional, Tuple

import numpy as np

import orbit.orbit_model_manager
import plugin_manager
from algorithm.algorithm_def import SHORTTERM, AlgorithmDef
from algorithm.data_trend import DataTrend
from algorithm.hybrid.hybrid_trend import HYBRID
from algorithm.training_output import TrainingOutputData
from algorithm.trend_node import TrendNode
from config.sdt_constants import DEFAULT, CURRENT
from orbit.geo_orbit_time import HOUR_IN_SECONDS
from config.sdt_config import session_period
from training.preprocessing.ex_zone import ExZone
from training.training_set import TrainingSet
from util import sdt_util

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.sdt_config as sdt_config
from sdtdb import sdt_db
import training.data_buffer as data_buffer
from algorithm.trend_tree import TrendTree
from algorithm.mnemonic_node import MnemonicNode
from dataio.sdt_data_input import SDTDataInput
from training.sdt_training import SDTTraining
from dataio.data_training_io import DataTrainingIO


# --- Constants ---
CONTEXT = "DataTrainingTask"


def _calculate_spc(node: MnemonicNode) -> None:
    """
    Calculates the system performance coefficient (SPC) for children of a node.
    
    SPC is a normalized ratio of the standard deviation of each mnemonic relative 
    to the average standard deviation across a large group (typically a subsystem).

    Args:
        node (MnemonicNode): The parent node whose children will have SPC calculated.
    """
    if node.is_leaf:
        return None
    num_child = len(node.children)
    if num_child > 30:
        sigma_array = np.array([trend.get_stddev() for trend in node.data_trend if trend.is_trended_check()])
        mean = sigma_array.mean()
        spc = sigma_array / mean
        for i, child in enumerate(node.children):
            if isinstance(child, MnemonicNode):
                child.data_trend[0].spc = spc[i]
    return None


def get_time_range_for_input_data(alg: AlgorithmDef, mnemonic_id: str) -> List[float]:
    """
    Determines the start and end timestamps for retrieving training data.
    
    The range covers the current session window but expands backward to include 
    enough pattern cycles required by the specific algorithm.

    Args:
        alg (AlgorithmDef): The algorithm definition.
        mnemonic_id (str): The identifier for the telemetry mnemonic.

    Returns:
        List[float]: A list containing [start_time, end_time] in Unix seconds.
    """
    s_time = data_buffer.session_start
    e_time = data_buffer.session_end
    num_patterns = alg.get_np()
    
    if num_patterns is not None and num_patterns > 0:
        default_model_time = data_buffer.get_default_model_time(alg)
        # Typically retrieve 2x the required patterns to handle overlaps and initialization
        s_time = data_buffer.session_end - 2.0 * num_patterns * default_model_time.get_model_period()
        
    if alg.get_name() == HYBRID:
        # For hybrid models, ensure the range covers all active state zones
        zones = data_buffer.get_state_zones(alg, mnemonic_id, CURRENT)
        if zones:
            for zone in zones:
                if zone:
                    z_array = zone.get_zones()
                    if z_array.size > 0:
                        s_time = min(z_array[0][0], s_time)
                        e_time = max(z_array[-1][1], e_time)
                        
    return [s_time, e_time]


def run_training_task(mnemonic_id: str, input_time: float, input_id: Optional[str], sat_id: str, session_time: float, is_manual_training: bool, session_type: int = SHORTTERM) -> List[TrainingOutputData]:
    """
    Standalone entry point for a single training task, intended for parallel execution.
    
    It re-initializes the satellite context (necessary for worker processes), 
    retrieves data, builds the hierarchical tree, and executes the training loop.

    Args:
        mnemonic_id (str): Subsystem or mnemonic ID to train.
        input_time (float): Reference time for the baseline model.
        input_id (Optional[str]): ID of the baseline model parameters.
        sat_id (str): Active satellite ID.
        session_time (float): Session reference timestamp.
        is_manual_training (bool): True if triggered manually.
        session_type (int): SHORTTERM (0) or LONGTERM (1).

    Returns:
        List[TrainingOutputData]: A list of training results for the mnemonics processed.
    """
    # Re-initialize globals for the current process
    sdt_config.set_sat_id(sat_id)
    data_buffer.set_session_time(session_time)
    data_buffer.session_type = session_type
    data_buffer.is_manual_training = is_manual_training
    
    logging.info(f"[{os.getpid()}] Starting training task for: {mnemonic_id}")

    data_input: SDTDataInput = plugin_manager.get_sdt_data_input("default")
    data_training_io = DataTrainingIO()
    sdt_training = SDTTraining(input_time, input_id, data_training_io)

    gap_threshold = sdt_config.get_float_property("GAPTHRESHOLD") or 0.4

    node = TrendTree.build_tree(mnemonic_id)
    if not node:
        logging.error(f"[{os.getpid()}] {CONTEXT}: Failed to build tree for {mnemonic_id}")
        return []

    training_output_list: List[TrainingOutputData] = []
    try:
        if sdt_db.is_subsystem(mnemonic_id):
            if node.children:
                for mn_node in node.children:
                    if isinstance(mn_node, MnemonicNode):
                        _train_single_mnemonic_node(mn_node, data_input, data_training_io, sdt_training, gap_threshold)
        else:
            if isinstance(node, MnemonicNode):
                _train_single_mnemonic_node(node, data_input, data_training_io, sdt_training, gap_threshold)
                
        training_output_list = generate_training_output_list(node)
        
        if not data_buffer.is_manual_training:
            data_training_io.save(training_output_list)

    except Exception as e:
        logging.error(f"[{os.getpid()}] {CONTEXT}: Training failed for {mnemonic_id}: {e}", exc_info=True)
        if isinstance(node, MnemonicNode):
            node.set_training_error(str(e))
    finally:
        if data_input:
            data_input.close()
        if data_training_io:
            data_training_io.close()

    return training_output_list

def _train_single_mnemonic_node(node: MnemonicNode, data_input: SDTDataInput, data_training_io: DataTrainingIO, sdt_training: SDTTraining, gap_threshold: float):
    """
    Manages the full training pipeline for a single telemetry point.
    
    Handles preprocessing steps including exclusion zone detection, model time 
    refinement, and data gap validation before invoking the training algorithm.

    Args:
        node (MnemonicNode): The node to train.
        data_input (SDTDataInput): Source for telemetry data.
        data_training_io (DataTrainingIO): Source for baseline models.
        sdt_training (SDTTraining): The training engine.
        gap_threshold (float): Maximum allowed data gap ratio.
    """
    if not data_input:
        return
        
    algorithm = AlgorithmDef(sdt_db.get_algorithm(node.name))
    _check_ex_zone(node, data_input, algorithm)
    _check_model_time(node, data_input, algorithm)
    
    training_set = _get_input_data(node, data_input, data_training_io, algorithm)

    training_period = algorithm.np * session_period if algorithm and algorithm.np > 1 else session_period

    if not training_set or len(training_set.raw) == 0:
        msg = f"No input data for {node.name}. Training skipped."
        node.set_training_error(msg)
        return

    sdt_util.sort_training_set(training_set)

    gap = sdt_util.check_data_gap(training_set, node.name, training_period) if data_buffer.session_type == SHORTTERM else 0.0
    
    if gap < gap_threshold:
        sdt_training.train(node, algorithm, training_set)
        if node.get_data_trends() is not None and len(node.data_trend) >= 30 and data_buffer.session_type == SHORTTERM:
            _calculate_spc(node)
    else:
        msg = f"Data gap exceeded ({gap:.2%}) for {node.name}."
        logging.error(f"[{os.getpid()}] {CONTEXT}: {msg}")
        node.set_training_error(msg)

def _get_input_data(node: MnemonicNode, data_input: SDTDataInput, data_training_io: DataTrainingIO, algorithm: AlgorithmDef) -> Optional[TrainingSet]:
    """
    Retrieves the necessary telemetry data for a mnemonic's training session.
    """
    try:
        if data_buffer.session_type == SHORTTERM:
            ranges = get_time_range_for_input_data(algorithm, node.name)
            return data_input.get_data(node.name, ranges[0], ranges[1])
        else:
            return data_training_io.get_stat_list(node.name, DEFAULT)
    except Exception as e:
        logging.error(f"[{os.getpid()}] {CONTEXT}: Data retrieval error for {node.name}: {e}")
        return None

def _check_ex_zone(mn_node: MnemonicNode, data_input: SDTDataInput, algorithm: AlgorithmDef):
    """
    Detects and initializes exclusion zones (e.g., maneuvers) required by the mnemonic's algorithm.
    """
    ex_zone_ids = set()
    ez_attr = algorithm.get_attribute("exzone")
    if ez_attr:
        ex_zone_ids.update(ez_attr.split("|"))
        
    if algorithm.get_name() == HYBRID:
        state_list = algorithm.al_type.state
        if state_list:
            for state in state_list:
                if state.flag is not None:
                    ex_zone_ids.update(re.split(r'[|$]', state.flag))
                    
    tpc_trend = algorithm.get_attribute("tpctrend")
    if tpc_trend == "yaw":
        ex_zone_ids.add("GNC_RO_YAW_FLIP_STATE")
        
    for zone_id in ex_zone_ids:
        if zone_id != "disjoint" and not data_buffer.ex_zone_exist(zone_id):
            training_set = _get_ex_zone_data(zone_id, data_input, algorithm)
            if training_set:
                ex_zone = ExZone.create_ex_zone(zone_id, training_set)
                if ex_zone:
                    data_buffer.add_ex_zone(ex_zone, zone_id)

def _get_ex_zone_data(mnemonic_id: str, data_input: SDTDataInput, algorithm: AlgorithmDef) -> Optional[TrainingSet]:
    """Retrieves data for an exclusion zone mnemonic."""
    default_model_time = orbit.orbit_model_manager.create_default_model_time(algorithm)
    start = data_buffer.session_time - 4.0 * default_model_time.get_model_period()
    end = data_buffer.session_end + 12 * HOUR_IN_SECONDS
    return data_input.get_data(mnemonic_id, start, end)

def _check_model_time(node: MnemonicNode, data_input: SDTDataInput, algorithm: AlgorithmDef):
    """
    Evaluates and refines pattern times if required by the mnemonic's algorithm.
    """
    model_time_id = algorithm.get_attribute('modeltime')
    if model_time_id and not data_buffer.is_model_time_exist(model_time_id):
        default_model_time = orbit.orbit_model_manager.create_default_model_time(algorithm)
        start = default_model_time.get_reference_time()
        end = data_buffer.session_end
        training_set = data_input.get_data(model_time_id, start, end)
        if training_set is not None:
            sample_period = sdt_db.get_mnemonic_type(node.name).frequency
            model_time = orbit.orbit_model_manager.get_model_time(algorithm, sample_period, training_set)
            data_buffer.add_model_time(model_time, model_time_id)

def generate_training_output_list(node: TrendNode) -> List[TrainingOutputData]:
    """
    Traverses the training tree and extracts serialized output objects for all mnemonics.
    """
    output_list: List[TrainingOutputData] = []
    if sdt_db.is_subsystem(node.name):
        for child in node.children:
            if isinstance(child, MnemonicNode):
                training_output = child.get_training_output()
                if training_output:
                    output_list.append(training_output)
    elif isinstance(node, MnemonicNode):
        output_list.append(node.get_training_output())
    return output_list


def main():
    """
    Main entry point for standalone script execution (testing).
    """
    parser = argparse.ArgumentParser(description="SDT Training Task Worker")
    parser.add_argument("group", help="Subsystem or Mnemonic ID")
    parser.add_argument("-t", "--type", type=int, default=0, choices=[0, 1])
    parser.add_argument("-s", "--satid", help="Satellite ID")
    parser.add_argument("--input_time", type=float, default=0.0)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    sdt_config.set_sat_id(args.satid or args.group)
    data_buffer.set_session_type(args.type)
    data_buffer.set_session_time(0)

    run_training_task(mnemonic_id=args.group, input_time=args.input_time, input_id=None, sat_id=args.satid, session_time=0, session_type=args.type, is_manual_training=False)

if __name__ == "__main__":
    main()
