import logging
import os
import sys
from typing import List

import plugin_manager
from algorithm.subsystem_output import SubsystemOutput
from posttraining.mnemonic_status import MnemonicStatus
from posttraining.sdt_analyzer import SDTAnalyzer

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdt_exception import SDTException
from posttraining.analyzer import Analyzer
from training import data_buffer

CONTEXT = "PostTrainingProcessingManager"


def post_training_analysis(data_list: List[SubsystemOutput]):
    """
    Coordinates the final post-training processing phase for a session.
    
    This function acts as the manager for:
    1. Executing the system-wide analysis and event clustering via an Analyzer.
    2. Generating operational health status records for all processed mnemonics.
    3. Persisting these status records to the data archive using pluggable I/O.

    Args:
        data_list (List[SubsystemOutput]): The complete set of training results 
            aggregated by subsystem.

    Raises:
        SDTException: If the analyzer cannot be initialized, if the analysis 
            process fails, or if there is an error during I/O operations.
    """
    if not data_list:
        logging.warning(f"{CONTEXT}: No data provided for post-training analysis.")
        return

    logging.info(f"{CONTEXT}: Starting Post Training Processing for session: {data_buffer.session_name}")

    # Initialize the analysis engine
    # In a fully pluggable system, this could be retrieved via plugin_manager
    analyzer: Analyzer = SDTAnalyzer()
    if not analyzer:
        raise SDTException("Failed to initialize analyzer for post-training processing.", CONTEXT)

    try:
        # Perform health analysis and generate status records
        status_list: List[MnemonicStatus] = analyzer.analyze(data_list)
    except Exception as e:
         raise SDTException(f"Critical failure during health analysis: {e}", CONTEXT)
    
    # Persist results to the archive
    sdt_status_io = plugin_manager.get_sdt_data_training_io("default")
    try:
        if sdt_status_io:
            logging.info(f"{CONTEXT}: Saving {len(status_list)} status records to archive.")
            sdt_status_io.write_status(status_list)
        else:
            logging.error(f"{CONTEXT}: I/O component unavailable. Status records will not be saved.")
    except SDTException as e:
         raise SDTException(f"IO Error while saving status records: {e}", CONTEXT)
    except Exception as e:
        logging.error(f"{CONTEXT}: Unexpected error during status archival: {e}", exc_info=True)
        raise SDTException("An unexpected error occurred while writing status into the SDT archive.", CONTEXT)
    finally:
        if sdt_status_io:
            sdt_status_io.close()
