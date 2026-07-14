import glob
import json
import logging
import os
from typing import List, Optional, Dict, Union

from config import sdt_config
from sdtdb.algorithm_type import AlgorithmType
from sdtdb.mnemonic_type import MnemonicType
from sdtdb.state_type import StateType
from sdtdb.subsystem_type import SubsystemType

# Constants
CONTEXT = "SDTDB"
JSON_DB_DIR = "json_db"
INDEX_FILE = "mnemonic_index.json"
LTTCOMPONENT = "LTTCOMPONENT"
DS = "ds"
RETRAIN = "retrain"
NORMALIZE = "norm"
MONITOR = "monitor"
DR = "dr"
INDR = "indr"
MIXED = "mixed"
NORMTYPE = "normtype"
TIMENAME = "timename"
SHIFTTYPE = "shifttype"
SHIFTPERIOD = "shiftperiod"
ALIAS = "alias"
TRAININGPARAM = "trainingparam"
TPCTREND = "tpctrend"
LONGTERM = 1
SHORTTERM = 0
MIXEDTYPE = 2 

# --- Caching ---
_dataset_cache: Dict[str, SubsystemType] = {}
_dataset_names_cache: Optional[List[str]] = None
_mnemonic_index_cache: Optional[Dict[str, List[str]]] = None


def _get_db_path() -> str:
    """
    Determines the path to the 'json_db' directory based on the active satellite ID.

    Returns:
        str: Absolute path to the directory containing JSON database files.
    """
    _db_dir = os.path.dirname(os.path.abspath(__file__))
    sat_id = sdt_config.sat_id
    _db_dir = os.path.join(os.path.dirname(_db_dir), "db", sat_id.lower() if sat_id else "", JSON_DB_DIR)
    
    if not os.path.exists(_db_dir):
        fallback_dir = os.path.join(os.path.dirname(_db_dir), "db", JSON_DB_DIR)
        if os.path.exists(fallback_dir):
            _db_dir = fallback_dir
            
    return _db_dir

def _load_mnemonic_index() -> Optional[Dict[str, List[str]]]:
    """
    Loads and caches the mnemonic index mapping subsystems to mnemonic names.

    Returns:
        Optional[Dict[str, List[str]]]: The index dictionary, or None if the 
            file is missing.
    """
    global _mnemonic_index_cache
    config_dir = sdt_config.config_dir
    if config_dir is None:
        json_db_dir = _get_db_path()
    else:
        json_db_dir = os.path.join(config_dir, JSON_DB_DIR)
        
    if _mnemonic_index_cache is not None:
        return _mnemonic_index_cache

    index_path = os.path.join(json_db_dir, INDEX_FILE)
    if not os.path.exists(index_path):
        logging.warning(f"{CONTEXT}: Mnemonic index file not found at {index_path}")
        return None

    try:
        with open(index_path, 'r') as f:
            _mnemonic_index_cache = json.load(f)
            return _mnemonic_index_cache
    except Exception as e:
        logging.error(f"{CONTEXT}: Error loading mnemonic index: {e}")
        return None

def get_subsystem_for_mnemonic(mnemonic_name: str) -> Optional[str]:
    """
    Identifies the subsystem (dataset) containing a specific mnemonic.

    Args:
        mnemonic_name (str): The name of the mnemonic.

    Returns:
        Optional[str]: The subsystem name if found, otherwise None.
    """
    index = _load_mnemonic_index()
    if not index:
        return None
        
    for subsystem, mnemonic_list in index.items():
        if mnemonic_name in mnemonic_list:
            return subsystem
            
    return None

def _parse_algorithm(alg_data: Union[Dict, List]) -> Optional[AlgorithmType]:
    """
    Parses a raw dictionary from JSON into an AlgorithmType object.

    Args:
        alg_data (Union[Dict, List]): The raw algorithm definition from JSON.

    Returns:
        Optional[AlgorithmType]: The parsed object, or None.
    """
    if not alg_data:
        return None
    if isinstance(alg_data, list):
        if not alg_data: return None
        alg_data = alg_data[0]
        
    # Parse States
    states = []
    states_data = alg_data.get('state', [])
    if isinstance(states_data, dict): states_data = [states_data]
    
    for s in states_data:
        states.append(StateType(
            name=s.get('name'),
            algorithm=s.get('algorithm'),
            flag=s.get('flag'),
            dim_pointer=int(s.get('dimpointer')) if s.get('dimpointer') else None,
            pad_factor=s.get('padfactor') if s.get('padfactor') else None
        ))
        
    # Parse Dimensions
    dims = alg_data.get('dim', [])
    if not isinstance(dims, list): dims = [dims]
    int_dims = [int(d) for d in dims]
    
    # Parse Attributes
    attrs = alg_data.get('attribute', [])
    if isinstance(attrs, dict): attrs = [attrs]

    return AlgorithmType(
        name=alg_data.get('name'),
        dim=int_dims,
        state=states,
        attribute=attrs,
        option=alg_data.get('option'),
        stat=alg_data.get('stat'),
        np=int(alg_data.get('np')) if alg_data.get('np') else None
    )

def get_dataset(dataset_name: str) -> Optional[SubsystemType]:
    """
    Loads and parses a complete subsystem (dataset) definition from its JSON file.

    Args:
        dataset_name (str): Name of the subsystem file (without .json).

    Returns:
        Optional[SubsystemType]: The populated subsystem object, or None if failed.
    """
    if dataset_name in _dataset_cache:
        return _dataset_cache[dataset_name]
        
    config_dir = sdt_config.config_dir
    json_db_dir = os.path.join(config_dir, JSON_DB_DIR) if config_dir else _get_db_path()

    json_path = os.path.join(json_db_dir, f"{dataset_name}.json")
    if not os.path.exists(json_path):
        logging.warning(f"{CONTEXT}: Dataset file not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
            mnemonics = []
            for mn_data in data.get('mnemonics', []):
                depends = mn_data.get('depends', [])
                
                sml = _parse_algorithm(mn_data.get('sml'))
                lml_data = mn_data.get('lml', [])
                if isinstance(lml_data, dict): lml_data = [lml_data]
                lml = []
                for l in lml_data:
                    alg = _parse_algorithm(l)
                    if alg: lml.append(alg)
                
                mnemonics.append(MnemonicType(
                    name=mn_data.get('name'),
                    frequency=float(mn_data.get('frequency', 1.0)),
                    warning_limit=float(mn_data.get('warninglimit', 0.0)),
                    error_limit=float(mn_data.get('errorlimit', 0.0)),
                    attributes=mn_data.get('attribute', []),
                    depends=depends,
                    sml=sml,
                    lml=lml
                ))
            
            dataset = SubsystemType(
                name=data.get('name'),
                satid=data.get('satid'),
                mnemonics=mnemonics,
            )
            
            _dataset_cache[dataset_name] = dataset
            return dataset
            
    except Exception as e:
        logging.error(f"{CONTEXT}: Error loading dataset '{dataset_name}': {e}")
        return None

def get_dataset_name_list() -> List[str]:
    """
    Returns a list of all defined subsystems by scanning the database directory.

    Returns:
        List[str]: List of subsystem names.
    """
    global _dataset_names_cache
    if _dataset_names_cache is not None:
        return _dataset_names_cache
        
    config_dir = sdt_config.config_dir
    if not config_dir:
        return []
        
    db_dir = os.path.join(config_dir, JSON_DB_DIR)
    if not os.path.exists(db_dir):
        return []

    json_files = glob.glob(os.path.join(db_dir, "*.json"))
    names = [os.path.splitext(os.path.basename(f))[0] for f in json_files]
    if "mnemonic_index" in names:
        names.remove("mnemonic_index")
    if "mnemonic_index_map" in names:
        names.remove("mnemonic_index_map")
        
    _dataset_names_cache = names
    return names

def is_subsystem(name: str) -> bool:
    """Checks if a name corresponds to a defined satellite subsystem."""
    return name in get_dataset_name_list()

def get_sat_id(subsystem_name: str) -> Optional[str]:
    """Retrieves the satellite ID associated with a subsystem."""
    dataset = get_dataset(subsystem_name)
    return dataset.satid if dataset else None

def get_mnemonic_type(mnemonic_id: str) -> Optional[MnemonicType]:
    """
    Retrieves the static configuration for a specific mnemonic.

    Args:
        mnemonic_id (str): Mnemonic name.

    Returns:
        Optional[MnemonicType]: The configuration object, or None.
    """
    subsystem_name = get_subsystem_for_mnemonic(mnemonic_id)
    dataset = get_dataset(subsystem_name)
    if dataset:
        for mn in dataset.mnemonics:
            if mn.name == mnemonic_id:
                return mn
    return None

def _get_attribute_value(attributes: List[str], attr_name: str) -> Optional[str]:
    """Helper to parse 'key|value' strings from attribute lists."""
    if not attributes:
        return None
        
    prefix = f"{attr_name}|"
    for attr in attributes:
        if isinstance(attr, str) and attr.startswith(prefix):
            return attr[len(prefix):]
    return None

def get_mnemonic_attribute(mnemonic_id: str, attr_name: str) -> Optional[str]:
    """Retrieves a specific attribute value for a mnemonic."""
    mn = get_mnemonic_type(mnemonic_id)
    return _get_attribute_value(mn.attributes, attr_name) if mn else None

def get_algorithm_attribute(mnemonic_id: str, attr_name: str, training_type: int = SHORTTERM) -> Optional[str]:
    """Retrieves an algorithm-specific attribute for a mnemonic."""
    alg = get_algorithm(mnemonic_id, training_type)
    return _get_attribute_value(alg.attribute, attr_name) if alg else None

def get_sample_period(mnemonic_id: str) -> float:
    """Returns the sampling frequency (period in seconds) for a mnemonic."""
    mn = get_mnemonic_type(mnemonic_id)
    return mn.frequency if mn else -1.0

def get_algorithm(mnemonic_id: str, training_type: int = SHORTTERM) -> Optional[AlgorithmType]:
    """Retrieves the algorithm configuration for a mnemonic and session type."""
    mn = get_mnemonic_type(mnemonic_id)
    if mn:
        return mn.sml if training_type == SHORTTERM else (mn.lml[0] if mn.lml else None)
    return None

def get_limits(mnemonic_id: str) -> Optional[List[float]]:
    """Returns the [warning, error] limit multipliers for a mnemonic."""
    mn = get_mnemonic_type(mnemonic_id)
    return [mn.warning_limit, mn.error_limit] if mn else None

def get_mnemonic_names(mnemonic_id: str) -> Optional[List[str]]:
    """Returns a mnemonic and its dependencies as a list of strings."""
    mn = get_mnemonic_type(mnemonic_id)
    if mn:
        names = [mn.name]
        if mn.depends:
            names.extend(mn.depends.split("|"))
        return names
    return None

def get_shift_type(mnemonic_id: str) -> Optional[str]:
    """Retrieves the pattern shift type attribute for a mnemonic."""
    return get_mnemonic_attribute(mnemonic_id, SHIFTTYPE)

def exist(name: str) -> bool:
    """Verifies if a name (mnemonic or subsystem) exists in the database."""
    if is_subsystem(name):
        return True
    return get_mnemonic_type(name) is not None
