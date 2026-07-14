import logging
import os
import sqlite3
import sys
from typing import List
from datetime import datetime, timezone

from dataio.sqlite import sqlite_utility
from dataio.sqlite.mnemonic_query import MnemonicQuerier
from posttraining.clustering.sdt_event_data import SDTEventData

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config.sdt_config as sdt_config

# --- Constants ---
CONTEXT = "EventIO"

class SQLiteEventIO:
    """
    Handles persistence of detected outlier events to an SQLite archive.
    
    This class manages a dedicated SQLite table to store processed event data, 
    allowing for historical analysis of anomalies and their correlations across 
    different operational states and mnemonics.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        Initializes the SQLiteEventIO with an active database connection.

        Args:
            connection (sqlite3.Connection): The SQLite connection object.
        """
        self.connection = connection
        self.event_period_limit = float(sdt_config.get_config_value("EVENTTIMELIMIT") or 3600.0)
        config_dir = sdt_config.config_dir
        mnemonic_query = MnemonicQuerier(config_dir)
        self.index = mnemonic_query.index
        self.create_event_table()

    def create_event_table(self):
        """Creates the event history table if it doesn't already exist."""
        try:
            self.connection.execute("PRAGMA page_size = 4096;")
            self.connection.execute("PRAGMA auto_vacuum = INCREMENTAL;")
            self.connection.execute("PRAGMA encoding = 'UTF-8';")
            cursor = self.connection.cursor()
            cursor.execute("""
                    CREATE TABLE if not exists sdt_event_history (
                        timetag INTEGER NOT NULL,
                        id  TEXT NOT NULL,
                        state TEXT NOT NULL,
                        period REAL NOT NULL,
                        value REAL NOT NULL,
                        PRIMARY KEY (timetag, id)
                    ) STRICT;
                    """)
            self.connection.commit()
        except sqlite3.Error as e:
            logging.error(f"[{CONTEXT}] Error creating table: {e}")

    def write_events(self, event_list: List[SDTEventData], cutoff: float):
        """
        Filters and persists a list of outlier events to the database.
        
        Only events that start after the specified cutoff time and meet 
        a minimum value threshold (e.g., 0.5) are saved.

        Args:
            event_list (List[SDTEventData]): The events to be archived.
            cutoff (float): The timestamp cutoff in Unix seconds.
        """
        if event_list:
            filtered_event_list = [event for event in event_list if event.event_start > cutoff and event.event_value > 0.5]
            if filtered_event_list:
                self.export_batch_data_to_file(filtered_event_list)
                event_ingest_list = self.get_event_tuple_list(filtered_event_list)
                if event_ingest_list:
                    self._write_batch(event_ingest_list)
        else:
            logging.info(f"{CONTEXT}: No event for the current training session")

    def _write_batch(self, batch_data: List[tuple]):
        """Helper to execute batch insertion into the sdt_event_history table."""
        try:
            cursor = self.connection.cursor()
            cursor.executemany("""
                INSERT INTO sdt_event_history (timetag, id, state, period, value) VALUES (?, ?, ?, ?, ?)
            """, batch_data)
            self.connection.commit()
        except sqlite3.Error as e:
            logging.error(f"[{CONTEXT}] Error writing batch: {e}")

    def export_batch_data_to_file(self, event_list: List[SDTEventData]):
        """
        Exports event data to a debug text file for diagnostic purposes.

        Args:
            event_list (List[SDTEventData]): The events to export.
        """
        try:
            config_dir = sdt_config.get_config_value("SQLITEDATAPATH")
            file_path = os.path.join(config_dir, sdt_config.sat_id, "event_batch.text")
            
            with open(file_path, "a") as f:
                for record in event_list:
                    f.write(f"{record}\n")
                f.write("\n")
            
            logging.info(f"[{CONTEXT}] Exported {len(event_list)} records to {file_path}")
        except Exception as e:
            logging.error(f"[{CONTEXT}] Failed to export batch data to file: {e}")

    def get_event_tuple_list(self, event_data_list: List[SDTEventData]) -> List[tuple]:
        """
        Serializes a list of event objects into tuples for database ingestion.

        Args:
            event_data_list (List[SDTEventData]): The source event objects.

        Returns:
            List[tuple]: List of (timetag_ms, id, state, period_s, value).
        """
        ingest_list = []
        for event_data in event_data_list:
            mnemonic_id = event_data.name
            event_value = event_data.event_value
            time_tag = int(event_data.event_start * 1000)
            period = event_data.event_end - event_data.event_start
            event_state = event_data.event_state
            ingest_list.append((time_tag, mnemonic_id, event_state, period, event_value))

        return ingest_list

    def get_event_history(self, start_time: float, end_time: float) -> List[SDTEventData]:
        """
        Retrieves historical event data from the database for a specific time range.

        Args:
            start_time (float): Start timestamp in Unix seconds.
            end_time (float): End timestamp in Unix seconds.

        Returns:
            List[SDTEventData]: The reconstructed list of event data objects.
        """
        event_data_list: List[SDTEventData] = []
        query_string = """
            SELECT timetag, id, state, period, value FROM sdt_event_history WHERE
                timetag >= ? AND
                timetag <= ?
        """
        try:
            start_int = int(start_time * 1000)
            end_int = int(end_time * 1000)
            cursor = self.connection.execute(query_string, (start_int, end_int))
            for timetag, mnemonic_id, state, period, value in cursor:
                sdt_event = SDTEventData(
                    name=mnemonic_id,
                    event_start=timetag / 1000.0,
                    event_end=timetag / 1000.0 + period,
                    event_value=value,
                    event_type="outlier",
                    event_state=state
                )
                event_data_list.append(sdt_event)
            cursor.close()
        except sqlite3.Error as e:
            logging.error(f"[{CONTEXT}] Error retrieving event history: {e}")

        return event_data_list

    def close(self):
        """Closes the database connection if it is open."""
        if self.connection:
            self.connection.close()
            self.connection = None
