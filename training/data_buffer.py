import gc
import logging
from typing import Dict, List, Optional, Tuple

import config.sdt_config as sdt_config
from algorithm.algorithm_def import AlgorithmDef
from algorithm.data_point import DataPoint
from algorithm.data_trend import DataTrend
from algorithm.subsystem_output import SubsystemOutput
from algorithm.training_output import TrainingOutputData
from algorithm.trend_node import TrendNode
from orbit import orbit_model_manager
from orbit.model_time import ModelTime
from posttraining.clustering.sdt_event import SDTEvent
from sdtdb import sdt_db
from training.preprocessing.ex_zone import ExZone
from util import time_util
from util.time_util import DAY_IN_SECONDS

# --- Caches ---
group_list: List[TrendNode] = []
"""List of high-level trend nodes for the current session."""

data_output_map: Dict[str, SubsystemOutput] = {}
"""Mapping of subsystem names to their training results."""

lt_trend_map: Dict[str, List[DataTrend]] = {}
"""Mapping for long-term trend storage."""

data_map: Dict[str, List[List[DataPoint]]] = {}
"""In-memory cache for raw telemetry data points."""

lt_data_map: Dict[str, List[DataPoint]] = {}
"""In-memory cache for long-term data points."""

model_time_map: Dict[str, ModelTime] = {}
"""Cache for refined model time objects, keyed by algorithm modeltime ID."""

exzone_map: Dict[str, ExZone] = {}
"""Cache for exclusion zones (e.g., maneuvers, dumps) keyed by zone identifier."""

default_list: List[Optional[ModelTime]] = [None] * 30
"""Cache for default model time objects indexed by number of patterns."""

event_map: Dict[str, List[SDTEvent]] = {}
"""Mapping of mnemonic IDs to their identified outlier events."""

training_error_map: Dict[str, str] = {}
"""Mapping of mnemonic IDs to error messages for failed training sessions."""

long_event_list: Optional[List[SDTEvent]] = None
"""List of events that span across multiple sessions."""

state_zone_map: Dict[str, List[Optional[List[ExZone]]]] = {}
"""Mapping of mnemonic IDs to lists of exclusion zones per state/session."""

# --- Session State Variables ---
ops_status: str = ""
"""Overall operational status string for the current session."""

ops_mode: int = 0
"""Current operation mode (0: DATATRAINING, 1: MONITORING)."""

session_start: float = 0.0
"""Start timestamp of the active training window (Unix seconds)."""

session_end: float = 0.0
"""End timestamp of the active training window (Unix seconds)."""

session_time: float = 0.0
"""Reference timestamp for the current session (usually the end point)."""

ecl_dur: float = -1.0
"""Duration of the active eclipse period in seconds."""

session_type: int = 0
"""Session type identifier (0: SHORTTERM, 1: LONGTERM)."""

session_name: str = ""
"""Human-readable name for the session (e.g., timestamp string)."""

is_manual_training: bool = False
"""Flag indicating if training was triggered manually."""

# --- Constants ---
CONTEXT = "DATABUFFER"
DATATRAINING = 0
MONITORING = 1
SHORTTERM = 0
LONGTERM = 1

def get_ops_mode() -> int:
    """Returns the current operational mode (TRAINING or MONITORING)."""
    return ops_mode

def set_ops_mode(mode: int):
    """Sets the current operational mode."""
    global ops_mode
    ops_mode = mode

def set_session_type(type_val: int):
    """
    Sets the training session type (Short or Long term). 
    Clears the buffer if the type changes.
    """
    global session_type
    if session_type != type_val:
        clear_buffer()
        session_type = type_val

def set_session_time(ref_time: float):
    """
    Initializes the session boundaries based on a reference time.
    
    If ref_time is 0, it uses default session boundaries from orbit manager.
    Otherwise, it calculates start and end points based on sdt_config settings.

    Args:
        ref_time (float): The target reference timestamp (Unix seconds).
    """
    global session_start, session_end, session_time, session_name
    if ref_time == 0:
        session_times = orbit_model_manager.get_session_time()
        session_start = session_times[0]
        session_end = session_times[1]
        session_time = session_times[1]
    else:
        session_time = ref_time
        if session_type == SHORTTERM:
            session_start = ref_time - sdt_config.session_period / 2
            session_end = ref_time + sdt_config.session_period / 2
        else:
            session_start = time_util.get_time_from_string(sdt_config.get_string_property("MISSIONSTART"))
            session_end = session_time
            
    session_name = time_util.get_simple_time_tag_from_seconds(session_time)
    clear_buffer()

def add_subsystem_output(subsystem_output: SubsystemOutput):
    """
    Adds a subsystem's training results to the buffer.

    Args:
        subsystem_output (SubsystemOutput): The results to cache.
    """
    global data_output_map
    data_output_map[subsystem_output.subsystem_name] = subsystem_output

def get_subsystem_output(subsystem_name: str) -> Optional[SubsystemOutput]:
    """Retrieves cached results for a subsystem."""
    return data_output_map.get(subsystem_name)

def get_training_delta() -> float:
    """Returns the configured training delta window in seconds."""
    return sdt_config.training_delta

def get_state_zones(alg: AlgorithmDef, mnemonic_id: str, session_index: int) -> Optional[List[ExZone]]:
    """
    Retrieves or initializes exclusion zones for a specific mnemonic and state.

    Args:
        alg (AlgorithmDef): Algorithm definition.
        mnemonic_id (str): Mnemonic name.
        session_index (int): PREV or CURRENT session indicator.

    Returns:
        Optional[List[ExZone]]: List of exclusion zones.
    """
    if mnemonic_id not in state_zone_map:
        state_zone_map[mnemonic_id] = [None, None]
    
    if state_zone_map[mnemonic_id][session_index] is None:
        from training.preprocessing.state_zones import StateZones
        state_zone_handler = StateZones(alg, mnemonic_id)
        state_zone_map[mnemonic_id][session_index] = state_zone_handler.init_state_zones(session_index)
        
    return state_zone_map[mnemonic_id][session_index]

def get_training_output_data(mnemonic_id: str) -> Optional[TrainingOutputData]:
    """
    Retrieves the training output (outliers, trends, status) for a mnemonic 
    by searching through cached subsystem outputs.

    Args:
        mnemonic_id (str): Mnemonic name.

    Returns:
        Optional[TrainingOutputData]: The output data, or None if not found.
    """
    sub_system_name = sdt_db.get_subsystem_for_mnemonic(mnemonic_id)
    if sub_system_name is not None and sub_system_name in data_output_map:
        sub_system_output = data_output_map[sub_system_name]
        for training_output in sub_system_output.mnemonic_output_list:
            if training_output.mnemonic_id == mnemonic_id:
                return training_output
    return None

def get_dt_tree_node(mnemonic_id: str) -> Optional[TrendNode]:
    """Retrieves a tree node from the buffer (implementation detail)."""
    return data_output_map.get(mnemonic_id)

def get_data_trend(mnemonic_id: str) -> Optional[List[DataTrend]]:
    """Retrieves the active DataTrend models for a mnemonic."""
    trend_node = get_dt_tree_node(mnemonic_id)
    from algorithm.mnemonic_node import MnemonicNode
    if isinstance(trend_node, MnemonicNode):
        return trend_node.get_data_trends()
    return None

def get_input_trend(mnemonic_id: str) -> Optional[List[DataTrend]]:
    """Retrieves the baseline/input DataTrend models for a mnemonic."""
    return get_data_trend(mnemonic_id)

def get_default_model_time(alg: AlgorithmDef) -> Optional[ModelTime]:
    """
    Retrieves or creates a default ModelTime object aligned with the orbital 
    pattern for an algorithm.

    Args:
        alg (AlgorithmDef): The algorithm definition.

    Returns:
        Optional[ModelTime]: The temporal context.
    """
    global default_list
    if default_list is None:
        default_list = [None] * 30

    index = max(0, alg.np - 1) if alg.np > 1 else 0
    
    if default_list[index] is None:
        default_list[index] = orbit_model_manager.create_default_model_time(alg)
        
    return default_list[index]

def get_session_model_time(alg: AlgorithmDef) -> Optional[ModelTime]:
    """
    Retrieves the specific ModelTime defined for an algorithm. 
    Falls back to default if no specific 'modeltime' attribute is set.

    Args:
        alg (AlgorithmDef): Algorithm definition.

    Returns:
        Optional[ModelTime]: The requested or default time context.
    """
    model_time_name = alg.get_attribute("modeltime")
    if model_time_name is None:
        return get_default_model_time(alg)
    return model_time_map.get(model_time_name)

def is_model_time_exist(model_time_name: str) -> bool:
    """Checks if a named model time context exists in the cache."""
    return model_time_name in model_time_map

def add_model_time(model_time: ModelTime, model_time_name: str):
    """Adds a refined model time context to the cache."""
    model_time_map[model_time_name] = model_time

def get_ex_zone(zone_id: str, start: float, end: float, padfactor: Optional[Tuple[float, float]] = None) -> Optional[ExZone]:
    """
    Retrieves and processes an exclusion zone.
    
    It supports merging multiple zones using the '|' separator, applying 
    padding factors, and filtering by time.

    Args:
        zone_id (str): Single ID or pipe-separated list of IDs.
        start (float): Start timestamp for filtering.
        end (float): End timestamp for filtering.
        padfactor (Optional[Tuple[float, float]]): Before/after padding durations.

    Returns:
        Optional[ExZone]: The processed exclusion zone.
    """
    zone_ids = zone_id.split('|')
    base_zone = exzone_map.get(zone_ids[0])
    if not base_zone:
        return None
        
    ezone = base_zone.copy()
    if len(zone_ids) > 1:
        for z_id in zone_ids[1:]:
            zone_to_merge = exzone_map.get(z_id)
            if zone_to_merge:
                ezone.merge(zone_to_merge, True)
    
    if ezone:
        if padfactor:
            ezone.add_pad(padfactor[0], padfactor[1])
        return ezone.filter_by_time(start, end)
    return None

def add_ex_zone(zone: ExZone, zone_id: str):
    """Adds an exclusion zone to the cache."""
    exzone_map[zone_id] = zone

def ex_zone_exist(zone_id: str) -> bool:
    """Checks if an exclusion zone ID is already cached."""
    return zone_id in exzone_map

def clear_buffer():
    """
    Resets all internal caches and session variables. 
    Triggers garbage collection to release memory.
    """
    global group_list, data_output_map, lt_trend_map, data_map, lt_data_map
    global model_time_map, exzone_map, default_list, event_map, training_error_map
    global long_event_list, state_zone_map, ecl_dur

    group_list.clear()
    data_output_map.clear()
    lt_trend_map.clear()
    data_map.clear()
    lt_data_map.clear()
    model_time_map.clear()
    exzone_map.clear()
    event_map.clear()
    training_error_map.clear()
    state_zone_map.clear()
    
    default_list = [None] * 30
    long_event_list = None
    ecl_dur = -1.0
    gc.collect()

def is_in_the_long_event_list_in_prev_session(mnemonic_id: str) -> bool:
    """Checks if a mnemonic was part of an ongoing long-period event."""
    if long_event_list is None:
        return False
    return any(mnemonic_id == event['name'] for event in long_event_list)

def get_ecl_duration() -> float:
    """
    Calculates and returns the duration of the solar eclipse in the current session.

    Returns:
        float: Duration in seconds, or 0 if no eclipse.
    """
    global ecl_dur
    if ecl_dur < 0:
        start = session_time
        end = session_time + DAY_IN_SECONDS
        ecl_id = sdt_config.get_string_property("ECLFLAG")
        if ecl_id:
            ex_zone = get_ex_zone(ecl_id, start, end, None)
            if ex_zone:
                # Assuming ex_zone supports indexing to get segments
                ecl_dur = ex_zone[0][1] - ex_zone[0][0]
            else:
                ecl_dur = 0
        else:
            ecl_dur = 0
    return ecl_dur
