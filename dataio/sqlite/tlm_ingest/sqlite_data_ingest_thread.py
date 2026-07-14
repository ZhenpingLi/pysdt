import logging
import os
import sqlite3
from typing import List

import numpy as np
import zstandard as zstd

from dataio.sqlite.mnemonic_query import MnemonicQuerier
from dataio.sqlite.tlm_ingest.sqlite_ingest_packet import SQLiteIngestPacket

CONTEXT = "NPP_DATA_INGEST"
DB_FILE = "npp_telemetry.db"

class SQLiteDataIngestThread:
    """
    Manages the ingestion of telemetry data into a local SQLite database.
    
    This class handles the connection to the SQLite database, ensures the 
    telemetry table exists, compresses incoming data using Zstandard, and 
    batches insertions for efficient storage. It maps mnemonic names to 
    integer IDs for optimized storage.
    """

    def __init__(self, config_path: str, sat_id: str, db_file: str = DB_FILE):
        """
        Initializes the SQLiteDataIngestThread.

        Args:
            config_path (str): The base directory for configuration and database files.
            sat_id (str): The satellite ID, used to name the database file.
            db_file (str): The name of the SQLite database file.
        """
        super().__init__()
        self.sat_id = sat_id
        self.read_status = True
        self.num_pt = 0
        mnemonic_id_map = MnemonicQuerier(config_path)
        self.index = mnemonic_id_map.index
        self.db_file = os.path.join(config_path, db_file)
        self.database_exists = os.path.exists(self.db_file)
        self.connection = None
        self.init_sqlite_connection()
        self._ensure_table_exists()
        self.ctx = zstd.ZstdCompressor(level=3)


    def init_sqlite_connection(self):
        """
        Establishes a connection to the SQLite database and applies performance PRAGMAs.
        """
        try:
            self.connection = sqlite3.connect(self.db_file)
            self.connection.execute("PRAGMA synchronous = OFF")
            self.connection.execute("PRAGMA cache_size = -64000")
            self.connection.execute("PRAGMA mmap_size = 30000000000")
        except sqlite3.Error as e:
            logging.error(f"[{CONTEXT}] Error connecting to SQLite database: {e}")

    def _ensure_table_exists(self):
        """
        Creates the 'telemetry' table if it does not exist.
        
        Configures the table with 'WITHOUT ROWID' for performance and 
        uses BLOBs for compressed telemetry values.
        """
        if not self.database_exists:
            try:
                self.connection.execute("PRAGMA auto_vacuum = NONE")
                self.connection.execute("PRAGMA page_size = 16384")
                self.connection.execute("PRAGMA mmap_size = 30000000000")
                cursor = self.connection.cursor()
                cursor.execute("""
                        CREATE TABLE if not exists telemetry (
                            timetag INTEGER NOT NULL,
                            id_int INTEGER NOT NULL,
                            value BLOB,
                            PRIMARY KEY (timetag, id_int)
                        ) WITHOUT ROWID
                    """)
                self.connection.commit()
            except sqlite3.Error as e:
                logging.error(f"[{CONTEXT}] Error creating table: {e}")

    def ingest_data(self, sql_data_list: List[SQLiteIngestPacket]):
        """
        Ingests a list of telemetry data packets into the SQLite database.
        
        Data values are compressed using Zstandard before being stored as BLOBs.
        Insertions are batched for efficiency.

        Args:
            sql_data_list (List[SQLiteIngestPacket]): A list of packets containing 
                telemetry data to be ingested.
        """
        try:
            cursor = self.connection.cursor()
            batch_data: List[tuple] = []
            for packet in sql_data_list:
                mnemonic_id = packet.mnemonic_id
                start_time = packet.times[0]
                values_np_array = np.array(packet.values)
                values_bytes = values_np_array.astype(np.float32).tobytes()
                value_blob = self.ctx.compress(values_bytes)
                
                if mnemonic_id in self.index:
                    id_int = self.index[mnemonic_id]
                    batch_data.append((start_time, id_int, value_blob))
                else:
                    logging.warning(f"[{CONTEXT}] Mnemonic '{mnemonic_id}' not found in index. Skipping ingestion.")

                if len(batch_data) >= 1000:
                    self._write_batch(cursor, batch_data)
                    self.connection.commit()
                    batch_data.clear()
                    self.num_pt += 1000
            
            if batch_data:
                self._write_batch(cursor, batch_data)
                self.connection.commit()
                self.num_pt += len(batch_data)
            logging.info(f"[{CONTEXT}] {self.num_pt} data points ingested.")
        except Exception as e:
            logging.error(f"[{CONTEXT}] Error in ingest_data: {e}", exc_info=True)


    def close_connection(self):
        """Closes the SQLite database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def _write_batch(self, cursor: sqlite3.Cursor, batch_data: List[tuple]):
        """
        Helper method to execute a batch insert of telemetry data.

        Args:
            cursor (sqlite3.Cursor): The SQLite cursor object.
            batch_data (List[tuple]): A list of tuples, each representing a row to insert.
        """
        try:
            cursor.executemany("""
                INSERT INTO telemetry (timetag, id_int, value) VALUES (?, ?, ?)
            """, batch_data)
        except sqlite3.Error as e:
            logging.error(f"[{CONTEXT}] Error writing batch: {e}", exc_info=True)

    def set_read_status(self, status: bool):
        """
        Sets the read status flag, typically used to signal the thread to stop.

        Args:
            status (bool): The new read status.
        """
        self.read_status = status
