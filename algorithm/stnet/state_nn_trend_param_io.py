import os
import logging
from typing import List, Optional
import numpy as np

# Assuming DataTrend and StateNNTrend will be in these locations
from algorithm.data_trend import DataTrend
# To avoid circular dependency, we'll use a forward reference or 'Any' if StateNNTrend imports this.
# For now, let's assume we can import it for type hinting.
from .state_nn_trend import StateNNTrend 

CONTEXT = "StateNNTrendParamIO"

def export_data_trend(trend: StateNNTrend):
    """
    Exports the given DataTrend object's parameters to a file.
    
    :param trend: The DataTrend object to export. It must be an instance of StateNNTrend
                  or a class that has a `get_net_file_name` method.
    """
    if not hasattr(trend, 'get_net_file_name'):
        logging.error(f"{CONTEXT}: Trend object does not have a 'get_net_file_name' method.")
        return

    file_path = trend.get_net_file_name()
    
    try:
        # Ensure the directory exists
        dir_name = os.path.dirname(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        with open(file_path, 'w') as writer:
            logging.info(f"{CONTEXT}: Exporting StateNNTrend for {trend.mnemonic_id} to {file_path}")
            
            model_params = trend.get_params()
            if model_params is None:
                logging.warning(f"{CONTEXT}: No model parameters to export for {trend.mnemonic_id}.")
                return
            
            # Convert parameters to strings and join with a pipe delimiter
            trend_string = "|".join(map(str, model_params))
            writer.write(trend_string + "\n")
            writer.close()
    except IOError as e:
        logging.error(f"{CONTEXT}: Error writing data trend for {trend.mnemonic_id}: {e}")
    except Exception as e:
        logging.error(f"{CONTEXT}: An unexpected error occurred during export: {e}")

def input_data_trend(trend: StateNNTrend) -> Optional[np.ndarray]:
    """
    Imports model parameters for a given DataTrend object from a file.
    
    :param trend: The DataTrend object for which to import parameters. It must be an instance
                  of StateNNTrend or have a `get_net_file_name` method.
    :return: The model parameters as a numpy array, or None if the file doesn't exist or an error occurs.
    """
    logging.info(f"{CONTEXT}: Importing StateNNTrend for {trend.mnemonic_id}")
    if not hasattr(trend, 'get_net_file_name'):
        logging.error(f"{CONTEXT}: Trend object does not have a 'get_net_file_name' method.")
        return None

    file_path = trend.get_net_file_name()
    
    if not os.path.exists(file_path):
        logging.info(f"{CONTEXT}: Trend parameter file not found at {file_path}. This may be expected for initial training.")
        return None
        
    try:
        with open(file_path, 'r') as reader:
            line = reader.readline()
            if not line:
                return None
                
            tokens = line.strip().split('|')
            # Convert tokens to a numpy array of floats
            model_params = np.array(tokens, dtype=float)
            reader.close()
            return model_params
            
    except IOError as e:
        logging.error(f"{CONTEXT}: Error reading the trend file {file_path}: {e}")
        return None
    except ValueError as e:
        logging.error(f"{CONTEXT}: Error parsing values in trend file {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"{CONTEXT}: An unexpected error occurred during import: {e}")
        return None
