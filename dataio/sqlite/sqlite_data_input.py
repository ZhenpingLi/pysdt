import logging
import os.path
import sqlite3
import traceback
from typing import Optional, Tuple, List

import numpy as np
import zstandard as zstd
from config import sdt_config
from dataio.sdt_data_input import SDTDataInput, _merge_data_list, commands
from dataio.sqlite.mnemonic_query import MnemonicQuerier
from sdtdb import sdt_db
from training.training_set import TrainingSet

BUFFER = 6060*1000 # 101 minutes in miliseconds.
CONTEXT = "DATAINPUT"

class SQLiteDataInput(SDTDataInput):
    """
    Implementation of SDTDataInput for reading telemetry data from SQLite.
    
    This class handles connecting to a satellite-specific SQLite database, 
    retrieving Zstandard-compressed telemetry BLOBs, and decompressing them 
    into NumPy arrays for training. It also supports merging multiple mnemonics 
    into a single TrainingSet for multi-input algorithms.
    """

    def __init__(self):
        """
        Initializes the SQLiteDataInput by loading the mnemonic index map 
        and initializing the Zstd decompressor.
        """
        self.connection = None
        self.sat_id = sdt_config.sat_id
        self.data_base_open = False
        mnemonic_id_map = MnemonicQuerier(sdt_config.config_dir)
        self.index = mnemonic_id_map.index
        self.dctx = zstd.ZstdDecompressor()
        self._init_connection(self.sat_id)
        self.orbit_mnemonic = sdt_config.get_config_value("ORBITMNEMONIC")

    def _init_connection(self, sat_id: str):
        """
        Establishes a connection to the SQLite database and configures performance PRAGMAs.

        Args:
            sat_id (str): The satellite ID, used to determine the database file path.
        """
        config_path = sdt_config.get_config_value("SQLITEDATAPATH")
        db_file_name = f"{sat_id}_telemetry.db"
        db_file_path = os.path.join(config_path, sat_id, db_file_name)
        logging.info(f"{CONTEXT}: Connecting to {db_file_path}")
        try:
            self.connection = sqlite3.connect(db_file_path)
            self.connection.execute("PRAGMA synchronous = NORMAL")
            self.connection.execute("PRAGMA cache_size = -256000") # 256MB cache
            self.connection.execute("PRAGMA mmap_size = 30000000000") # 30GB MMAP
            self.data_base_open = True
        except sqlite3.Error as e:
            logging.error(f"{CONTEXT}: Failed to connect to SQLite: {e}")

    def _check_connection(self):
        """Ensures the database connection is active, re-initializing if necessary."""
        if not self.connection or not self.data_base_open:
            self._init_connection(self.sat_id)

    def get_data(self, mnemonic_id: str, start: float, end: float) -> Optional[TrainingSet]:
        """
        Main entry point to retrieve and process a full TrainingSet for a mnemonic.
        
        This method retrieves the primary mnemonic data and any dependent data 
        defined in the database, merges them, and applies any required 
        preprocessing functions (e.g., 'pad', 'bin').

        Args:
            mnemonic_id (str): The logical path/ID of the mnemonic.
            start (float): Start timestamp (Unix seconds).
            end (float): End timestamp (Unix seconds).

        Returns:
            Optional[TrainingSet]: The populated TrainingSet, or None if no data 
                was found.
        """
        if self.orbit_mnemonic is not None and mnemonic_id == self.orbit_mnemonic:
            mn_names = [mnemonic_id]
        else:
            mn_names = sdt_db.get_mnemonic_names(mnemonic_id)
            
        if not mn_names:
            return None
            
        input_data, output_data = self.get_mnemonic_data(mn_names[0], start, end)
        if input_data is None or output_data is None:
            return None
            
        training_set = TrainingSet(
            mnemonic_id=mnemonic_id,
            inputs=input_data,
            raw=output_data,
            outputs=output_data,
            dqf=np.ones(len(input_data), dtype=np.int8)
        )

        if self.orbit_mnemonic is not None and mnemonic_id != self.orbit_mnemonic:
            if len(mn_names) > 1:
                column_times = []
                column_data = []
                for name in mn_names[1:]:
                    _times, _data = self.get_mnemonic_data(name, start, end)
                    if _times is not None and _data is not None:
                        column_times.append(_times.flatten())
                        column_data.append(_data)
                _merge_data_list(training_set, column_times, column_data)

            _function = sdt_db.get_algorithm_attribute(mnemonic_id, "function")
            if _function:
                _function = _function.strip()
                if _function in commands:
                    commands[_function](training_set)
                else:
                    logging.warning(f"{CONTEXT}: Function '{_function}' not found in commands list.")

        return training_set

    def get_mnemonic_data(self, mnemonic_id: str, start: float, end: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieves raw, decompressed data for a specific mnemonic from the database.

        Args:
            mnemonic_id (str): The mnemonic name.
            start (float): Start timestamp (Unix seconds).
            end (float): End timestamp (Unix seconds).

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: A tuple of 
                (times, values) arrays, or (None, None) if not found.
        """
        self._check_connection()
        if not self.connection or mnemonic_id not in self.index:
            return None, None
            
        int_id = self.index[mnemonic_id]
        start_int = int(start * 1000) - BUFFER
        end_int = int(end * 1000)
        
        query = """SELECT timetag, value FROM telemetry WHERE 
                 id_int = ? AND 
                 timetag >= ? AND 
                 timetag <= ?
                 ORDER BY timetag ASC"""
                 
        all_times = []
        all_values = []
        try:
            cursor = self.connection.execute(query, (int_id, start_int, end_int))
            t_start_ms = int(start * 1000)

            for start_t, blob in cursor:
                # Decompress BLOB to NumPy array
                raw_values = np.frombuffer(self.dctx.decompress(blob), dtype=np.float32)

                # Reconstruct timestamps (ms)
                times = start_t + (np.arange(len(raw_values)) * 1000)
                times_float = times.astype(np.float64) / 1000.0
                
                # Filter to exact range
                mask = (times >= t_start_ms) & (times <= end_int)
                if np.any(mask):
                    all_times.append(times_float[mask])
                    all_values.append(raw_values[mask])
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"{CONTEXT} error retrieving data {mnemonic_id}: {e}\n{error_details}")

        if all_times and all_values:
            time_array = np.concatenate(all_times).reshape(-1, 1)
            value_array = np.concatenate(all_values)
            return time_array, value_array
        else:
            return None, None

    def close(self):
        """Closes the SQLite connection and marks the database as closed."""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.data_base_open = False
        logging.info(f"{CONTEXT}: SQLite Connection client closed.")
