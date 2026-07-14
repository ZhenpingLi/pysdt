import logging
import os
import sqlite3
import sys
import traceback
from contextlib import closing
from sqlite3 import Cursor
from typing import List, Optional, Tuple, Any

import numpy as np

import plugin_manager
from algorithm.algorithm_data import AlgorithmData
from algorithm.algorithm_def import AlgorithmDef
from algorithm.hybrid_state_data import HybridStateData
from algorithm.single_state_data import SingleStateData
from config import sdt_config
from config.sdt_constants import DEFAULT, FBNN, STNET
from dataio.sqlite.mnemonic_query import MnemonicQuerier
from dataio.sqlite.sqlite_utility import get_data_training_io_connection, get_data_training_output_db_path
from posttraining.clustering.sdt_event_data import SDTEventData
from posttraining.mnemonic_status import MnemonicStatus
from sdt_exception import SDTException
from sdtdb import sdt_db
from training.training_worker import HYBRID

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataio.sdt_data_io import SDTDataTrainingIO
from algorithm.data_point import DataPoint
from util import time_util as tu

# --- Constants ---
CONTEXT = "SQLiteDataIO"
SDTOUTPUT = "sdt_output"
OPSTATUS = "opstatus"

# DATA FIELD CONSTANTS
TRENDFIELD = 0        # Model parameters (weights/coefficients)
PATTERNTIMEFIELD = 1  # Pattern period and timestamps
STATFIELD = 2         # Statistical parameters (mean, max, min, sigma)
SOFIELD = 3           # Scale and offset parameters
OPSTATUSFIELD = 4     # Operational health status metrics


def _get_stat_array(stat_array: List[DataPoint], time: float, ref_time: float, pattern_period: float, state: str) -> Optional[np.ndarray]:
    """
    Retrieves the statistical data array for a specific time and state.
    
    Args:
        stat_array (List[DataPoint]): The list of statistical data points.
        time (float): The target timestamp.
        ref_time (float): The reference time of the session.
        pattern_period (float): The pattern cycle duration.
        state (str): The operational state name.

    Returns:
        Optional[np.ndarray]: The statistical data array, or None.
    """
    if state == DEFAULT:
        _index = int(round((time - ref_time) / pattern_period))
        if len(stat_array) > _index >= 0:
            if stat_array[_index] is None:
                stat_array[_index] = DataPoint(ref_time, np.zeros(4))
            return stat_array[_index].data
        else:
            return None
    else:
        for index in range(len(stat_array)):
            if stat_array[index] is None:
                stat_array[index] = DataPoint(ref_time, np.zeros(4))
                return stat_array[index].data
            elif time == stat_array[index].time:
                return stat_array[index].data
    return None


def create_algorithm_data(mnemonic_id: str, state: str) -> Optional[AlgorithmData]:
    """
    Factory function to initialize a SingleStateData object from the database definition.
    
    Args:
        mnemonic_id (str): The mnemonic identifier.
        state (str): The operational state.

    Returns:
        SingleStateData: The initialized (but empty) algorithm data container.
    """
    algorithm = AlgorithmDef(sdt_db.get_algorithm(mnemonic_id))
    algorithm_name = algorithm.get_name()
    num_pattern_in_training = sdt_config.get_int_property("NUMPATTERNINTRAINING")
    if algorithm.np:
        num_pattern_in_training = algorithm.np
    state_type = algorithm.get_state(state)
    if state_type is not None:
        algorithm_name = state_type.algorithm
    return SingleStateData(
        mnemonic_id=mnemonic_id,
        alg_name=algorithm_name,
        ref_time=0,
        pattern_period=0,
        pattern_times=None,
        stat_list=None,
        scale_offset_list=None,
        sigma=0.0,
        tpc=1.0,
        params=None,
        num_pattern_in_training=num_pattern_in_training,
        state=state
    )


def get_param_number(algorithm_data: AlgorithmData) -> Tuple[int, int]:
    """
    Retrieves the dimensionality of parameters for a specific algorithm.
    
    Args:
        algorithm_data (AlgorithmData): The algorithm data object.

    Returns:
        Tuple[int, int]: A tuple of (trained_params_dim, total_model_params_dim).
    """
    from algorithm.data_trend import DataTrend
    data_trend : DataTrend = plugin_manager.get_data_trend(algorithm_data.alg_name, algorithm_data.mnemonic_id)
    if data_trend:
        return data_trend.get_param_dim(), data_trend.get_model_param_dim()
    else:
        return 0, 0


def get_data_point(sigma_list: List[DataPoint], time: float, dim: int) -> DataPoint:
    """
    Retrieves or creates a DataPoint object for a specific timestamp in a list.
    
    Args:
        sigma_list (List[DataPoint]): The list to search.
        time (float): The timestamp.
        dim (int): The dimensionality of the data array.

    Returns:
        DataPoint: The existing or newly created DataPoint.
    """
    if not sigma_list:
        data_point = DataPoint(time, np.zeros(dim))
        sigma_list.append(data_point)
        return data_point
    for data_point in sigma_list:
        if data_point.time == time:
            return data_point
    data_point = DataPoint(time, np.zeros(dim))
    sigma_list.append(data_point)
    return data_point


def export_batch_data_to_file(event_list: List[SDTEventData]):
    """
    Exports a list of events to a debug text file.
    
    Args:
        event_list (List[SDTEventData]): The events to export.
    """
    try:
        config_dir = sdt_config.get_config_value("SQLITEDATAPATH")
        file_path = os.path.join(config_dir, sdt_config.sat_id, "event_batch.text")
        with open(file_path, "a") as f:
            for record in event_list:
                f.write(f"{record}\n")
        logging.info(f"[{CONTEXT}] Exported {len(event_list)} records to {file_path}")
    except Exception as e:
        logging.error(f"[{CONTEXT}] Failed to export batch data to file: {e}")


def check_wal_health(threshold_mb=50) -> bool:
    """
    Checks if the SQLite Write-Ahead Log (WAL) file size exceeds a healthy threshold.
    
    Args:
        threshold_mb (int): The size threshold in Megabytes.

    Returns:
        bool: True if the WAL file should be truncated, False otherwise.
    """
    db_path = get_data_training_output_db_path()
    wal_path = f"{db_path}-wal"
    if os.path.exists(wal_path):
        wal_size_mb = os.path.getsize(wal_path) / (1024 * 1024)
        if wal_size_mb > threshold_mb:
            return True
    return False


class SQLiteSDTDataIO(SDTDataTrainingIO):
    """
    SQLite implementation of SDTDataTrainingIO for persisting training archives.
    
    This class handles the conversion of complex algorithm state objects into 
    binary BLOBs stored in SQLite, allowing for efficient retrieval and 
    long-term tracking of model performance.
    """

    def __init__(self):
        """Initializes the SQLite I/O component and loads the mnemonic ID map."""
        mnemonic_query = MnemonicQuerier(sdt_config.config_dir)
        self.index = mnemonic_query.index
        self.connection = None

    def _check_connection(self):
        """Ensures an active connection to the SQLite database."""
        if self.connection is None:
            self.connection = get_data_training_io_connection()

    def write_data_trend(self, algorithm_data_list: List[AlgorithmData]):
        """
        Persists a list of trained algorithm models to the SQLite database.

        Args:
            algorithm_data_list (List[AlgorithmData]): The data objects to save.
        """
        if not algorithm_data_list:
            return
        mnemonic_id = algorithm_data_list[0].mnemonic_id
        try:
            for algorithm_data in algorithm_data_list:
                trend_lines = []
                if algorithm_data.alg_name == HYBRID:
                    h_trend: HybridStateData = algorithm_data
                    for sub_trend in h_trend.data_trend_list:
                        trend_lines.extend(self.generate_data_trend_tuple(sub_trend))
                else:
                    trend_lines.extend(self.generate_data_trend_tuple(algorithm_data))
                if trend_lines:
                    self._write_batch(trend_lines)
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"{CONTEXT}: Error writing data for {mnemonic_id}: {e}\n{error_details}")
            raise SDTException(f"Error writing data for {mnemonic_id}: {e}")

    def generate_data_trend_tuple(self, algorithm_data: SingleStateData) -> List[tuple]:
        """
        Serializes a SingleStateData object into a list of tuples for SQLite insertion.

        Args:
            algorithm_data (SingleStateData): The model parameters and stats.

        Returns:
            List[tuple]: List of (timetag, id_int, data_field, state, value_blob).
        """
        tuple_list = []
        time_tag = int(algorithm_data.ref_time * 1000)
        int_id = self.index[algorithm_data.mnemonic_id]
        
        # Serialize Model Parameters
        params = algorithm_data.params
        param_dim = 1 + len(params) if len(params)>0 else 1
        output_params = [0.0] * param_dim
        output_params[0] = algorithm_data.sigma
        if len(params) > 0 and params is not None:
            output_params[1:] = params
        trend_blob = np.array(output_params, dtype=np.float32).tobytes()
        tuple_list.append((time_tag, int_id, TRENDFIELD, algorithm_data.state, trend_blob))
        
        # Serialize Pattern Times
        pattern_times = algorithm_data.pattern_times
        if pattern_times is not None:
            pattern_time_array = [0.0] * len(pattern_times)
            pattern_time_array[0] = algorithm_data.pattern_period
            pattern_time_array[1:] = pattern_times[1:] - time_tag
            pattern_time_blob = np.array(pattern_time_array, dtype=np.float32).tobytes()
            tuple_list.append((time_tag, int_id, PATTERNTIMEFIELD, algorithm_data.state, pattern_time_blob))
        
        # Serialize Statistics
        if algorithm_data.stat_list:
            for data_point in algorithm_data.stat_list:
                dp_time_tag = int(data_point.time * 1000)
                stat_blob = np.array(data_point.data, dtype=np.float32).tobytes()
                tuple_list.append((dp_time_tag, int_id, STATFIELD, algorithm_data.state, stat_blob))
        
        # Serialize Scale/Offsets
        if algorithm_data.scale_offset_list:
            for data_point in algorithm_data.scale_offset_list:
                so_time_tag = int(data_point.time * 1000)
                so_blob = np.array(data_point.data, dtype=np.float32).tobytes()
                tuple_list.append((so_time_tag, int_id, SOFIELD, algorithm_data.state, so_blob))

        return tuple_list

    def _write_batch(self, batch_data: List[tuple]):
        """Helper method to perform batch INSERT OR REPLACE operations."""
        try:
            with closing(sqlite3.connect(get_data_training_output_db_path())) as conn:
                with conn:
                    cursor = conn.cursor()
                    cursor.executemany("""
                        INSERT OR REPLACE INTO sdt_output (timetag, id_int, data_field, state, value) VALUES (?, ?, ?, ?, ?)
                        """, batch_data)
                    conn.commit()
        except sqlite3.Error as e:
            logging.error(f"[{CONTEXT}] Error writing batch: {e}")

    def execute_query(self, mnemonic_id: str, start: float, end: float, field: int, state: str) -> Cursor:
        """
        Helper to execute a standardized SELECT query for training data.

        Args:
            mnemonic_id (str): The mnemonic name.
            start (float): Start timestamp.
            end (float): End timestamp.
            field (int): The data field constant.
            state (str): The operational state.

        Returns:
            Cursor: The SQLite cursor for the results.
        """
        self._check_connection()
        int_id = self.index[mnemonic_id]
        if state is None:
            state = DEFAULT
        start_int = int(start * 1000)
        end_int = int(end * 1000)
        query = """
            SELECT timetag, value FROM sdt_output WHERE 
            id_int = ? AND
            data_field = ? AND
            state = ? AND 
            timetag >= ? AND 
            timetag <= ?
            ORDER BY timetag ASC
        """
        params = (int_id, field, state, start_int, end_int)
        return self.connection.execute(query, params)

    def get_data_trend(self, mnemonic_id: str, start: float, end: float, state: str) -> Optional[AlgorithmData]:
        """
        Retrieves and reconstructs a SingleStateData object from the archive.

        Args:
            mnemonic_id (str): ID of the mnemonic.
            start (float): Start timestamp.
            end (float): End timestamp.
            state (str): Operational state.

        Returns:
            Optional[AlgorithmData]: The reconstructed model data, or None.
        """
        logging.info(f"{CONTEXT}: get data trend for {mnemonic_id} from {tu.get_time_tag_from_seconds(start)} to {tu.get_time_tag_from_seconds(end)}")
        algorithm_data: SingleStateData = create_algorithm_data(mnemonic_id, state)
        
        # Load Model Parameters
        cursor = self.execute_query(mnemonic_id, start, end, TRENDFIELD, state)
        if cursor:
            self.process_model_data(cursor, algorithm_data)
        cursor.close()
        
        # Load Pattern Times
        cursor = self.execute_query(mnemonic_id, start, end, PATTERNTIMEFIELD, state)
        if cursor:
            self.process_pattern_times(cursor, algorithm_data)
        cursor.close()
        
        if algorithm_data.pattern_times is not None:
            # Load Statistics
            stat_end = algorithm_data.pattern_times[-1] - 300
            cursor = self.execute_query(mnemonic_id, start, stat_end, STATFIELD, state)
            if cursor:
                algorithm_data.stat_list = self.process_stat(cursor)
            cursor.close()
            
            # Load Scale/Offsets for specific algorithms
            if algorithm_data.alg_name == FBNN or algorithm_data.alg_name == STNET:
                cursor = self.execute_query(mnemonic_id, start, end, SOFIELD, state)
                if cursor:
                    algorithm_data.scale_offset_list = self.process_stat(cursor)
                cursor.close()
                
        return algorithm_data if algorithm_data.params is not None else None

    def process_model_data(self, cursor: Cursor, algorithm_data: SingleStateData):
        """Parses model parameter BLOBs into the algorithm data object."""
        for timetag, values in cursor:
            model_params = np.frombuffer(values, dtype=np.float32)
            algorithm_data.sigma = float(model_params[0])
            algorithm_data.params = model_params[1:].tolist()
            algorithm_data.ref_time = float(timetag) / 1000.0

    def process_pattern_times(self, cursor: Cursor, algorithm_data: SingleStateData):
        """Parses pattern time BLOBs into the algorithm data object."""
        for timetag, values in cursor:
            pattern_array = np.frombuffer(values, dtype=np.float32)
            algorithm_data.pattern_period = float(pattern_array[0])
            pattern_times = [0.0] * len(pattern_array)
            pattern_times[0] = float(timetag) / 1000.0
            pattern_times[1:] = (pattern_array[1:] + timetag) / 1000.0
            algorithm_data.pattern_times = np.array(pattern_times)

    def process_stat(self, cursor: Cursor) -> List[DataPoint]:
        """Parses generic statistic BLOBs into a list of DataPoints."""
        stat_array = []
        for timetag, values in cursor:
            data_array = np.frombuffer(values, dtype=np.float32)
            stat_array.append(DataPoint(time=float(timetag) / 1000.0, data=data_array.tolist()))
        return stat_array

    def write_status(self, status_list: List[MnemonicStatus]):
        """
        Writes the overall operational health status to the archive.

        Args:
            status_list (List[MnemonicStatus]): List of health status records.
        """
        if not status_list:
            return
        point_list = []
        for status in status_list:
            int_id = self.index[status.mnemonic_id]
            status_blob = np.array(status.status_array, dtype=np.float32).tobytes()
            point_list.append((int(status.time * 1000), int_id, OPSTATUSFIELD, status.state_id, status_blob))
        self._write_batch(point_list)

    def get_data_stats(self, mnemonic_id: str, start: float, end: float, state: str) -> Optional[List[DataPoint]]:
        """Retrieves historical stats from the database."""
        cursor = self.execute_query(mnemonic_id, start, end, STATFIELD, state)
        if cursor:
            return self.process_stat(cursor)
        return None

    def get_sigma(self, mnemonic_id: str, start: float, end: float, state: str) -> Optional[List[DataPoint]]:
        """Retrieves history of standard deviations (sigma)."""
        sigma_list = []
        cursor = self.execute_query(mnemonic_id, start, end, TRENDFIELD, state)
        if cursor:
            for timetag, values in cursor:
                model_params = np.frombuffer(values, dtype=np.float32)
                sigma_list.append(DataPoint(time=float(timetag) / 1000.0, data=[float(model_params[0])]))
        return sigma_list if sigma_list else None

    def get_event_history(self, start_time: float, end_time: float) -> Optional[List[SDTEventData]]:
        """Retrieves outlier event history from the database."""
        event_data_list: List[SDTEventData] = []
        query_string = """
            SELECT timetag, id, state, period, value FROM sdt_event_history WHERE
                timetag >= ? AND
                timetag <= ?
            """
        try:
            with closing(sqlite3.connect(get_data_training_output_db_path())) as conn:
                with conn:
                    start_int = int(start_time * 1000)
                    end_int = int(end_time * 1000)
                    cursor = conn.execute(query_string, (start_int, end_int))
                    for timetag, mnemonic_id, state, period, value in cursor:
                        sdt_event = SDTEventData(
                            name=mnemonic_id,
                            event_start=timetag / 1000.0,
                            event_end=(timetag + (period * 1000)) / 1000.0,
                            event_value=value,
                            event_type="outlier",
                            event_state=state
                        )
                        event_data_list.append(sdt_event)
                    cursor.close()
        except sqlite3.Error as e:
            logging.error(f"[{CONTEXT}] Error retrieving event history: {e}")
        return event_data_list

    def write_events(self, event_list: List[SDTEventData], cutoff: float):
        """Persists a list of outlier events to the database."""
        if event_list:
            filtered_event_list = [event for event in event_list if event.event_start > cutoff and event.event_value > 0.5]
            if filtered_event_list:
                export_batch_data_to_file(filtered_event_list)
                event_ingest_list = self.get_event_tuple_list(filtered_event_list)
                if event_ingest_list:
                    self._write_event_batch(event_ingest_list)

    def _write_event_batch(self, batch_data: List[tuple]):
        """Executes batch insert for event data."""
        try:
            with closing(sqlite3.connect(get_data_training_output_db_path())) as conn:
                with conn:
                    cursor = conn.cursor()
                    cursor.executemany("""
                        INSERT INTO sdt_event_history (timetag, id, state, period, value) VALUES (?, ?, ?, ?, ?)
                        """, batch_data)
                    conn.commit()
                    cursor.close()
        except sqlite3.Error as e:
            logging.error(f"[{CONTEXT}] Error writing batch: {e}")

    def get_event_tuple_list(self, event_data_list: List[SDTEventData]) -> List[tuple]:
        """Converts SDTEventData list into SQLite tuples."""
        ingest_list = []
        for event_data in event_data_list:
            time_tag = int(event_data.event_start * 1000)
            period = event_data.event_end - event_data.event_start
            ingest_list.append((time_tag, event_data.name, event_data.event_state, period, event_data.event_value))
        return ingest_list

    def close(self):
        """Closes the connection and performs WAL maintenance if necessary."""
        if self.connection:
            if check_wal_health():
                self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            self.connection.close()
            self.connection = None
        logging.info(f"{CONTEXT}: SQLite Connection client closed.")
