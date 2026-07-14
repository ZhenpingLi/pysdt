import logging
import os
import sqlite3

from algorithm.algorithm_data import LONGTERM
from config import sdt_config
from sdt_exception import SDTException
from training import data_buffer


def get_data_training_io_connection() -> sqlite3.Connection:
    """
    Establishes and configures a connection to the data training output database.
    
    This function handles the creation of the database file if it doesn't 
    exist and applies optimized SQLite PRAGMAs (WAL mode, memory mapping) 
    for high-performance data training archives. It also ensures necessary 
    tables are initialized.

    Returns:
        sqlite3.Connection: An active and configured SQLite connection object.

    Raises:
        SDTException: If the connection or database initialization fails.
    """
    try:
        sqlite_db_path = get_data_training_output_db_path()
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
        
        table_exists = os.path.exists(sqlite_db_path)
        sqlite_connection = sqlite3.connect(sqlite_db_path)
        
        sqlite_connection.execute("PRAGMA journal_mode = WAL")
        sqlite_connection.execute("PRAGMA busy_timeout = 5000")
        sqlite_connection.execute("PRAGMA synchronous = NORMAL")
        sqlite_connection.execute("PRAGMA cache_size = -256000")
        sqlite_connection.execute("PRAGMA mmap_size = 10000000000")
        
        create_training_output_table(sqlite_connection, table_exists)
        create_event_table(sqlite_connection, table_exists)
        
        return sqlite_connection
    except Exception as e:
        raise SDTException(f"Failed to initialize SQLite training archive: {e}")

def get_data_training_output_db_path() -> str:
    """
    Constructs the absolute file system path for the training output database.
    
    The path is determined by the satellite ID and the session type 
    (SHORTTERM vs LONGTERM).

    Returns:
        str: The full path to the .db file.
    """
    config_dir = sdt_config.get_config_value("SQLITEDATAPATH")
    sqlite_db_file = f"{sdt_config.sat_id}_sdt_output.db"
    if data_buffer.session_type == LONGTERM:
        sqlite_db_file = f"{sdt_config.sat_id}_lt_sdt_output.db"
    sqlite_db_path = os.path.join(config_dir, sdt_config.sat_id, sqlite_db_file)
    return sqlite_db_path

def create_event_table(event_connection: sqlite3.Connection, table_exists: bool):
    """
    Creates the 'sdt_event_history' table if it is missing from the database.

    Args:
        event_connection (sqlite3.Connection): Active SQLite connection.
        table_exists (bool): Flag indicating if the database file already exists.
    """
    if not table_exists:
        try:
            event_connection.execute("PRAGMA page_size = 4096;")
            event_connection.execute("PRAGMA auto_vacuum = INCREMENTAL;")
            event_connection.execute("PRAGMA encoding = 'UTF-8';")
            cursor = event_connection.cursor()
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
            event_connection.commit()
        except sqlite3.Error as e:
            logging.error(f"Error creating event history table: {e}")

def create_training_output_table(connection: sqlite3.Connection, table_exists: bool):
    """
    Creates the 'sdt_output' table using optimized performance settings.
    
    Uses 'WITHOUT ROWID' for storage efficiency and configures large page 
    sizes for massive model parameter BLOBs.

    Args:
        connection (sqlite3.Connection): Active SQLite connection.
        table_exists (bool): Flag indicating if the database file already exists.
    """
    if not table_exists:
        try:
            connection.execute("PRAGMA auto_vacuum = NONE")
            connection.execute("PRAGMA page_size = 16384")
            connection.execute("PRAGMA mmap_size = 10000000000")
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute("PRAGMA busy_timeout = 5000;")
            cursor = connection.cursor()
            cursor.execute("""
                        CREATE TABLE if not exists sdt_output (
                            timetag INTEGER NOT NULL,
                            id_int INTEGER NOT NULL,
                            data_field INTEGER NOT NULL,
                            state STRING NOT NULL,                            
                            value BLOB,
                            PRIMARY KEY (timetag, id_int, data_field)
                        ) WITHOUT ROWID
                    """)
            connection.commit()
        except sqlite3.Error as e:
            logging.error(f"Error creating sdt_output table: {e}")
