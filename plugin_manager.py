import importlib
import json
import logging
import os
from typing import Dict, Optional, Any

from algorithm.algorithm_data import AlgorithmData
from algorithm.algorithm_factory import AlgorithmFactory
from algorithm.data_trend import DataTrend
from config import sdt_config
from config.sdt_constants import SHORTTERM
from dataio.sdt_data_input import SDTDataInput
from dataio.sdt_data_io import SDTDataTrainingIO
from sdt_exception import SDTException
from training.training_worker import TrainingWorker

# --- Module-level State ---
_plugin_cache: Dict[str, Any] = {}
"""Cache for instantiated plugin objects to ensure singleton-like behavior."""

_class_registry: Optional[Dict] = None
"""In-memory dictionary mapping plugin names to their fully qualified Python paths."""

CONTEXT = "PluginManager"


def _load_registry():
    """
    Loads the class registry from 'class_registry.json' into memory.
    
    This registry file is located in the satellite-specific configuration 
    directory and provides the mapping between logical plugin names and 
    concrete Python implementations.
    """
    global _class_registry
    if _class_registry is not None:
        return
        
    current_dir = sdt_config.config_dir
    if not current_dir:
        logging.error(f"{CONTEXT}: Global configuration not initialized. Cannot load registry.")
        return
        
    registry_path = os.path.join(current_dir, 'class_registry.json')
    try:
        with open(registry_path, 'r') as f:
            _class_registry = json.load(f)
            logging.info(f"{CONTEXT}: Class registry loaded successfully from {registry_path}.")
    except FileNotFoundError:
        logging.error(f"{CONTEXT}: class_registry.json not found at {registry_path}")
        _class_registry = {}
    except json.JSONDecodeError as e:
        logging.error(f"{CONTEXT}: Error decoding JSON from class_registry.json: {e}")
        _class_registry = {}


def _get_plugin(plugin_type: str, name: str) -> Optional[Any]:
    """
    Generic core function to dynamically import and instantiate a plugin.
    
    It uses the registry to find the Python path, performs a dynamic import, 
    and caches the resulting instance for future use.

    Args:
        plugin_type (str): The category of the plugin (e.g., 'AlgorithmFactory').
        name (str): The specific identifier for the plugin (e.g., 'fbnn').

    Returns:
        Optional[Any]: The instantiated plugin object, or None if loading failed.
    """
    _load_registry()

    cache_key = f"{plugin_type}:{name}"
    if cache_key in _plugin_cache:
        return _plugin_cache[cache_key]

    if not _class_registry or plugin_type not in _class_registry:
        logging.error(f"{CONTEXT}: Plugin type '{plugin_type}' not found in class registry.")
        return None

    plugin_info = _class_registry[plugin_type].get(name)
    if not plugin_info:
        logging.warning(f"{CONTEXT}: Plugin '{name}' of type '{plugin_type}' not found in registry.")
        return None

    try:
        # Expected format: "package.module.ClassName"
        module_path, class_name = plugin_info.rsplit('.', 1)
        
        module = importlib.import_module(module_path)
        plugin_class = getattr(module, class_name)
        
        # Instantiate the class
        instance = plugin_class()
        
        _plugin_cache[cache_key] = instance
        logging.info(f"{CONTEXT}: Successfully loaded '{name}' for type '{plugin_type}'.")
        return instance
        
    except (ImportError, AttributeError, TypeError, ValueError) as e:
        logging.error(f"{CONTEXT}: Failed to load plugin '{name}' from '{plugin_info}': {e}")
        return None


def get_algorithm_factory(name: str) -> Optional[AlgorithmFactory]:
    """
    Retrieves the factory object for a specific training algorithm.

    Args:
        name (str): The name of the algorithm (e.g., 'ridgefe').

    Returns:
        Optional[AlgorithmFactory]: The algorithm factory instance.
    """
    return _get_plugin("AlgorithmFactory", name)


def get_sdt_data_input(name: str = "default") -> Optional[SDTDataInput]:
    """
    Retrieves the component responsible for reading raw telemetry data.

    Args:
        name (str): The identifier for the data input plugin. Defaults to 'default'.

    Returns:
        Optional[SDTDataInput]: The data input instance.
    """
    return _get_plugin("SDTDataInput", name)


def get_sdt_data_training_io(name: str = "default") -> Optional[SDTDataTrainingIO]:
    """
    Retrieves the component responsible for archiving training results.

    Args:
        name (str): The identifier for the training I/O plugin. Defaults to 'default'.

    Returns:
        Optional[SDTDataTrainingIO]: The training I/O instance.
    """
    return _get_plugin("SDTDataTrainingIO", name)


def get_data_trend(alg_name: str, mnemonic_id: str, session_type: int = SHORTTERM) -> DataTrend:
    """
    High-level convenience method to create a new DataTrend object via its factory.

    Args:
        alg_name (str): The name of the algorithm.
        mnemonic_id (str): The identifier for the telemetry mnemonic.
        session_type (int): The current training session type.

    Returns:
        DataTrend: A fresh, uninitialized DataTrend object.

    Raises:
        SDTException: If the algorithm factory cannot be loaded.
    """
    factory = get_algorithm_factory(alg_name)
    if factory:
        return factory.get_data_trend(mnemonic_id=mnemonic_id)
    else:
        raise SDTException(f"Algorithm factory '{alg_name}' not defined in registry.", CONTEXT)


def get_training_worker(alg_name: str) -> TrainingWorker:
    """
    Convenience method to retrieve the appropriate TrainingWorker for an algorithm.

    Args:
        alg_name (str): The name of the algorithm.

    Returns:
        TrainingWorker: The worker instance.

    Raises:
        SDTException: If the algorithm factory cannot be loaded.
    """
    factory = get_algorithm_factory(alg_name)
    if factory:
        return factory.get_training_worker()
    else:
        raise SDTException(f"Algorithm factory '{alg_name}' not defined in registry.", CONTEXT)


def get_data_trend_from_output(algorithm_data: AlgorithmData) -> DataTrend:
    """
    Reconstructs a live DataTrend object from a serialized AlgorithmData object.

    Args:
        algorithm_data (AlgorithmData): The serialized model data.

    Returns:
        DataTrend: The reconstructed model object with its parameters populated.

    Raises:
        SDTException: If the algorithm factory cannot be loaded.
    """
    factory = get_algorithm_factory(algorithm_data.alg_name)
    if factory:
        data_trend: DataTrend = factory.get_data_trend(algorithm_data.mnemonic_id)
        data_trend.set_algorithm_data(algorithm_data)
        return data_trend
    else:
        raise SDTException(f"Algorithm factory '{algorithm_data.alg_name}' not defined.", CONTEXT)
