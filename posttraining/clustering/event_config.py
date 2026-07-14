from config import sdt_config
from config.sdt_constants import DAY_IN_SECONDS

# Global Configuration Parameters for Event Clustering
event_threshold = 2.0
"""The minimum value for an outlier cluster to be considered an event."""

event_history_limit = 30.0 * DAY_IN_SECONDS
"""The look-back window (in seconds) for retrieving historical event data."""

event_warning_threshold = 3.0
"""The threshold value above which an event is categorized as a WARNING."""

event_error_threshold = 5.0
"""The threshold value above which an event is categorized as an ERROR."""

oc_join_time: float = 5.0
"""The maximum time gap (in seconds) to merge contiguous outlier clusters."""

SCALE: float = 100.0
"""The scaling factor used to normalize outlier values for clustering."""

event_debug = False
"""Flag to enable verbose logging for the clustering process."""

min_pts = 3
"""The 'min_samples' parameter for the DBSCAN clustering algorithm."""

event_period_limit = 3600.0
"""The maximum duration (in seconds) for an event; longer events are separated."""

cluster_threshold = 5.0
"""The minimum density threshold for DBSCAN to form a cluster."""

eps = 0.9
"""The 'eps' (radius) parameter for the DBSCAN clustering algorithm."""

event_join_time = 5.0
"""The maximum time gap (in seconds) to merge hierarchical SDTEvents."""

def init_event_config():
    """
    Initializes the global event configuration parameters from system settings.
    
    This function reads values from the active satellite configuration (via 
    sdt_config) and updates the global constants. It should be called at the 
    beginning of a training or analysis session.
    """
    global event_threshold, event_history_limit, event_warning_threshold, event_error_threshold, oc_join_time
    global SCALE, event_debug, min_pts, event_period_limit, cluster_threshold, eps, event_join_time
    
    event_threshold = float(sdt_config.get_config_value("EVENTTHRESHOLD") or 2.0)
    event_history_limit = float(sdt_config.get_config_value("EVENTHISTORYLIMIT") or 30.0) * DAY_IN_SECONDS
    event_warning_threshold = float(sdt_config.get_config_value("EVENTWARNING") or 3.0)
    event_error_threshold = float(sdt_config.get_config_value("EVENTERROR") or 5.0)
    oc_join_time = float(sdt_config.get_config_value("OCJOINTIME") or 5.0)
    SCALE = float(sdt_config.get_config_value("OUTLIERSCALE") or 100.0)
    event_debug = sdt_config.get_config_value("EVENTDEBUG") == "YES"
    min_pts = sdt_config.get_int_property("MINPOINTS") or 3
    event_period_limit = float(sdt_config.get_config_value("EVENTTIMELIMIT") or 3600)
    cluster_threshold = float(sdt_config.get_config_value("CLUSTERTHRESHOLD") or 5.0)
    eps = float(sdt_config.get_config_value("EVENTEPSVALUE") or 0.9)
    event_join_time = float(sdt_config.get_config_value("EVENTJOINTIME") or 5.0)
