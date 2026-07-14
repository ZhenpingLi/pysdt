import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from influxdb_client.client.write.point import Point

import plugin_manager
from algorithm.single_state_data import SingleStateData
from dataio.influxdb import influxdb_util
from posttraining.mnemonic_status import MnemonicStatus
from training.training_worker import DEFAULT
from util import time_util
from util.time_util import HOUR_IN_SECONDS

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.sdt_config as sc
import training.data_buffer as db

# --- Constants ---
# These would typically be managed in a more central config
OPSSTATUS = "opsstatus"
TREND = "trend"
LTTREND = "lttrend"
MLMODEL = "mlmodel"
OUTLIER = "outlier"
STDDEV = "stddev"
EVENT = "event"

# --- Placeholder for AIMSTlmMap ---

def get_flux_query_string(mnemonic_id: str, start: float, end: float, field: str, state: Optional[str]=DEFAULT, postfix : str = None) -> str:
    """
    Generates a Flux query to retrieve trend or model data.
    """
    # InfluxDB requires RFC3339 format with 'Z' for UTC
    start_s = datetime.fromtimestamp(start, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_s = datetime.fromtimestamp(end, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    if state is None:
        state = "default"
    # Use a placeholder for the bucket name, which should come from config
    bucket = influxdb_util.get_influx_bucket(influxdb_util.MODELIO)
    measurement = mnemonic_id
    
    # Build filter predicates
    filters = [
        f'r._measurement == "{measurement}"',
        f'r._field == "{field}"'
    ]
    if postfix is not None:
        filters.append(f'r.postfix == "{postfix}"')
    if state:
        filters.append(f'r.state == "{state}"')
        
    filter_str = " and ".join(filters)

    flux_query = f"""from(bucket: "{bucket}")|> range(start: {start_s}, stop: {end_s}) 
    |> filter(fn: (r) => {filter_str})"""
    return flux_query.strip()

def get_sigma_query_string(mnemonic_id: str, start: float, end: float, state: Optional[str]) -> str:
    """
    Generates a Flux query to retrieve sigma values from the operational status bucket.
    """
    start_s = datetime.fromtimestamp(start, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_s = datetime.fromtimestamp(end, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    bucket = influxdb_util.get_influx_bucket(influxdb_util.MODELIO)

    measurement = mnemonic_id
    
    state = state or "default"
    
    filter_str = f'r._measurement == "{measurement}" and r.state == "{state}" and r._field == "trend" and r.index=="0"'

    flux_query = f"""
        from(bucket: "{bucket}")
        |> range(start: {start_s}, stop: {end_s})
        |> filter(fn: (r) => {filter_str})
    """
    return flux_query.strip()

def get_model_points_for_line_protocol(trends: List[SingleStateData]) -> List[Point]:
    """
    Generates InfluxDB line protocol strings for the ML model's output.
    """
    # This function is highly complex and depends on many other components
    # (DataModelUtility, ExZone, etc.) that are not fully defined.
    # A simplified placeholder is provided.
    
    if not trends:
        return []

    trend = trends[0]
    data_trend = plugin_manager.get_data_trend(trend.alg_name, trend.mnemonic_id)# Use the first trend for metadata
    measurement_tag = trend.mnemonic_id
    data_trend.set_algorithm_data(trend)
    
    cutoff_time = db.session_end - (db.get_training_delta() * HOUR_IN_SECONDS)
    sample_time = float(sc.get_config_value("OUTPUTSAMPLETIME") or 2.0)
    
    current_time = cutoff_time
    point_list : List[Point] = []
    while current_time < db.session_end:
        # In a real implementation, you would find the correct trend for the current_time
        # and generate the input vector `t` for it.
        # For this placeholder, we'll just use a dummy value.
        
        # t = model_utility.get_data_model_input(...)
        # data_value = trend.get_trend_value(t)
        
        data_value = data_trend.get_trend_value([current_time]) # Placeholder value
        dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
        if not np.isnan(data_value):
            point = Point(measurement_tag).field("mlmodel", data_value).time(dt)
            point_list.append(point)
        current_time += sample_time
        
    return point_list

def get_status_points_for_line_protocol(status: MnemonicStatus) -> Optional[Point]:
    """
    Generates InfluxDB line protocol strings for a DataQualityMetrics point.
    """
    if not status:
        return None
    state = status.state_id
    measurement_tag = status.mnemonic_id
    time = status.time
    status_string : List[str] = [str(s) for s in status.status_array]
    values = "|".join(status_string)
    dt = datetime.fromtimestamp(time, tz=timezone.utc)
    influxdb_point = Point(measurement_tag).tag("state", state).field("status", values).time(dt)
    return influxdb_point

def get_trend_points_for_line_protocol(algorithm_data: SingleStateData) -> List[Point]:
    """
    Generates InfluxDB line protocol strings for a DataTrend object's parameters and stats.
    """
    if not algorithm_data:
        return []
        
    lines = []
    params : Optional[List[float]] = algorithm_data.params
    pattern_times = algorithm_data.pattern_times
    model_params : List[float] = [
        algorithm_data.sigma,
        *params,
        *pattern_times[1:]
    ]
    ref_time = algorithm_data.ref_time
    measurement_tag = algorithm_data.mnemonic_id
    state = algorithm_data.state
    dt = datetime.fromtimestamp(ref_time, tz=timezone.utc)
    logging.info(f"InfluxDBDataIO: Writing trend for {measurement_tag} at the state {state} with the reference time: {time_util.get_time_tag_from_seconds(ref_time)}")
    
    # Write model parameters

    for i, p_val in enumerate(model_params):
        if not np.isnan(p_val):
            point = Point(measurement_tag).tag("state", state).tag("index", str(i)).field(TREND, float(p_val)).time(dt)
            lines.append(point)
    # Write statistics
    if algorithm_data.stat_list:
        for dp in algorithm_data.stat_list:
            if dp and dp.time > db.session_time:
                dt = datetime.fromtimestamp(dp.time, tz=timezone.utc)
                for i, s_vale in (enumerate(dp.data)):
                    point = Point(measurement_tag).tag("state", state).tag("index", str(i)).field("stat", s_vale).time(dt)
                    lines.append(point)
    if algorithm_data.scale_offset_list:
        for dp in algorithm_data.scale_offset_list:
            if dp and dp.time > db.session_time:
                dt = datetime.fromtimestamp(dp.time, tz=timezone.utc)
                for j, s_val in enumerate(dp.data):
                    point = Point(measurement_tag).tag("state", state).tag("index", str(j)).field("scale_offset", s_val).time(dt)
                    lines.append(point)
    return lines

def get_outlier_points_for_line_protocol(outlier_list)-> List[str]:
    if outlier_list and len(outlier_list) > 0:
        lines=[]
        for outlier in outlier_list:
            time= int (outlier.time * 1e3)
            mn_name = outlier['mnemonic_id']
            lines.append(f"""outlier, mn_name="{mn_name}" value={outlier.normalized_value} {time}""")
        return lines
    else:
        return None
