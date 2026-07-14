import logging

from influxdb_client import InfluxDBClient

import config.sdt_config as sdt_config

# Constants for client modes
DATAINPUT = 0
MODELIO = 1
OPSTATUS = 2

# Module-level cache for buckets
_data_input_bucket = None
_sdt_bucket = None
_ops_status_bucket = None

CONTEXT = "InfluxDBUtil"

def get_influx_client(mode: int = DATAINPUT) -> InfluxDBClient:
    """
    Creates and returns an InfluxDB client based on the specified mode.

    :param mode: The mode to determine which configuration to use.
                 0 (DATAINPUT): For data input (e.g., from CASSIE).
                 1 (AIMSINPUT): For AIMS trending data output.
                 2 (OPSTATUS): For operational status data.
    :return: An initialized InfluxDBClient.
    """
    tokens = None
    host_name = None
    
    if mode == MODELIO or mode == OPSTATUS:
        tokens = sdt_config.get_config_value("INFLUXDB/MODELOUTPUT/TOKENS")
        host_name = sdt_config.get_config_value("INFLUXDB/MODELOUTPUT/HOSTNAME")
    elif mode == DATAINPUT:
        tokens = sdt_config.get_config_value("INFLUXDB/DATAINPUT/TOKENS")
        host_name = sdt_config.get_config_value("INFLUXDB/DATAINPUT/HOSTNAME")

    if not host_name:
        print(f"ERROR: {CONTEXT}: Hostname not configured for mode {mode}")
        return None
        
    if not host_name.startswith("http"):
        host_name = f"http://{host_name}"

    if not tokens:
        print(f"ERROR: {CONTEXT}: Tokens not configured for mode {mode}")
        return None

    # In Python, OkHttpClient is configured directly on the options
    #options = InfluxDBClientOptions.builder() \
    #    .url(host_name) \
    #    .token(tokens) \
    #    .org(get_influx_org(mode)) \
    #    .read_timeout(300_000) \
    #    .write_timeout(300_000) \
    #    .connect_timeout(300_000) \
    #    .build()
    #logging.info(f"{CONTEXT}, connecting to {host_name} with tokens '{tokens}' and org {get_influx_org(mode)}")
    client = InfluxDBClient(url=host_name, token=tokens, org=get_influx_org(mode), enable_gzip=True)
    return client

def get_influx_org(mode: int) -> str:
    """
    Retrieves the InfluxDB organization based on the mode.
    """
    if mode == MODELIO or mode == OPSTATUS:
        return sdt_config.get_config_value("INFLUXDB/MODELOUTPUT/ORG")
    elif mode == DATAINPUT:
        return sdt_config.get_config_value("INFLUXDB/DATAINPUT/ORG")
    return None

def get_influx_bucket(mode: int) -> str:
    """
    Retrieves the InfluxDB bucket name based on the mode, with caching.
    """
    global _data_input_bucket, _sdt_bucket, _ops_status_bucket

    if mode == DATAINPUT:
        if _data_input_bucket is None:
            # In a multi-satellite context, you might need to select the bucket differently
            # For now, we'll assume a simple case or that the config provides a single value.
            bucket_key = f"INFLUXDB/DATAINPUT/BUCKET/{sdt_config.sat_id}"
            logging.info(f"{CONTEXT}, using bucket key {bucket_key}")
            buckets = sdt_config.get_config_value(bucket_key)
            if isinstance(buckets, list) and buckets:
                # Logic to select the correct bucket based on sat_id would go here
                # For now, just picking the first one as a placeholder.
                _data_input_bucket = buckets[0] 
            else:
                _data_input_bucket = buckets # If it's just a string
        logging.info(f"{CONTEXT}, using bucket {_data_input_bucket}")
        return _data_input_bucket
        
    elif mode == MODELIO:
        if _sdt_bucket is None:
            _init_sdt_bucket()
        return _sdt_bucket
        
    elif mode == OPSTATUS:
        if _ops_status_bucket is None:
            _init_sdt_bucket()
        return _ops_status_bucket
        
    else: # Default case
        if _sdt_bucket is None:
            _init_sdt_bucket()
        return _sdt_bucket

def _init_sdt_bucket():
    """
    Initializes the AIMS and OPS status bucket names from configuration.
    """
    global _sdt_bucket, _ops_status_bucket
    bucket_key = f"INFLUXDB/MODELOUTPUT/BUCKET/{sdt_config.sat_id}"
    b_names = sdt_config.get_config_value(bucket_key)
    
    if isinstance(b_names, list) and b_names:
        # Assuming a list of "BUCKET|SAT_ID" strings
        sat_id = sdt_config.sat_id
        if not sat_id:
            logging.warning(f"{CONTEXT}: sdt_config.sat_id not set. Defaulting to first bucket.")
            # Default to the first entry if sat_id is not set
            selected_bucket_str = b_names[0]
        else:
            # Find the bucket that ends with the current satellite ID
            found_bucket = next((b for b in b_names if b.endswith(f"|{sat_id}")), None)
            if found_bucket:
                selected_bucket_str = found_bucket
            else:
                logging.warning(f"WARNING: {CONTEXT}: No bucket found for sat_id '{sat_id}'. Defaulting to first.")
                selected_bucket_str = b_names[0]

        # The bucket name is the part before the pipe
        _sdt_bucket = selected_bucket_str.split('|')[0]
        _ops_status_bucket = _sdt_bucket # Assuming they are the same in this new structure
        
    elif isinstance(b_names, str):
        # Handle the old format "BUCKET1|BUCKET2"
        _tokens = b_names.split('|')
        if len(_tokens) == 2:
            _sdt_bucket = _tokens[0]
            _ops_status_bucket = _tokens[1]
        else:
            print(f"WARNING: {CONTEXT}: Needs to specify both AIMS bucket and OPS status bucket in 'INFLUXDB/BUCKET'")
            _sdt_bucket = _tokens[0]
            _ops_status_bucket = _tokens[0]
    else:
        print(f"ERROR: {CONTEXT}: 'INFLUXDB/MODELOUTPUT/BUCKET' is not configured correctly.")
        _sdt_bucket = ""
        _ops_status_bucket = ""
