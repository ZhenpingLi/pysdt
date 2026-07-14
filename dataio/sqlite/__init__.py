"""
SDT SQLite I/O Implementation.

This sub-package provides concrete implementations of the data input and 
training I/O interfaces using SQLite as the storage backend.

Key Features:
*   **Storage Efficiency**: Telemetry data is compressed using Zstandard 
    before being stored as binary BLOBs.
*   **Optimized Performance**: Uses SQLite features such as WAL mode, 
    memory mapping (MMAP), and large page sizes to handle large-scale 
    telemetry archives.
*   **Binary Parameter Storage**: Model parameters and statistical metrics 
    are serialized into binary format for high-speed read/write operations.
*   **Mnemonic ID Mapping**: Includes a utility to map long mnemonic strings 
    to fixed-length integer IDs, significantly reducing index size and 
    query time.

The package includes the :class:`.SQLiteDataInput` for retrieval and 
:class:`.SQLiteSDTDataIO` for archival of results.
"""
