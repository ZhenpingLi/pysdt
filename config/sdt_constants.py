import logging

# --- Version ---
VERSION = "5.1.0"
"""Current version of the AIMS-SDT application."""

# --- Event and Status Levels ---
WARNING = 0
"""Generic warning level indicator."""
ERROR = 1
"""Generic error level indicator."""
NORMAL = 2
"""Generic normal status indicator."""
INFO = 2
"""Generic information level indicator (often same as NORMAL)."""
TYPESTRING = ["W", "E", "I"]
"""String representations for WARNING, ERROR, INFO."""

MAXITER = 2500
"""Maximum iterations for some iterative algorithms (e.g., neural networks)."""
DEFAULTITER = 1400
"""Default iterations for some iterative algorithms."""

PREV = 0
"""Index for previous data/state."""
CURRENT = 1
"""Index for current data/state."""

# --- Configuration Keys ---
VERBOSE = "verbose"
"""Configuration key for verbose logging."""
VERB = "v"
"""Short configuration key for verbose logging."""

# --- Statistical Indices and Names ---
MAX_INDEX = 0
"""Index for maximum value in a statistical array."""
MIN_INDEX = 1
"""Index for minimum value in a statistical array."""
MEAN_INDEX = 2
"""Index for mean value in a statistical array."""
SIGMA_INDEX = 3
"""Index for standard deviation (sigma) in a statistical array."""
STATINDEX = [MAX_INDEX, MIN_INDEX, MEAN_INDEX, SIGMA_INDEX]
"""List of all statistical indices."""

MAX = "max"
"""String identifier for maximum statistic."""
MIN = "min"
"""String identifier for minimum statistic."""
MEAN = "mean"
"""String identifier for mean statistic."""
SIGMA = "sigma"
"""String identifier for standard deviation statistic."""
MNPOSTFIX = [MAX, MIN, MEAN, SIGMA]
"""List of common statistical postfixes."""

# --- Trend and Limit Identifiers ---
WLIMIT = "wlimit"
"""Identifier for warning limit trend."""
ELIMIT = "elimit"
WELIMIT = "welimit"
"""Identifier for error limit trend."""
"""Identifier for both warning and error limits trend."""
TREND = "trend"
"""Identifier for the primary trend value."""
STDDEV = "stddev"
"""Identifier for standard deviation."""
ORDER = "order"
"""Identifier for algorithm order (e.g., polynomial degree)."""
NONE = "none"
"""Generic identifier for no specific type."""
EXZONE = "exzone"
"""Identifier for exclusion zone attribute."""
MINSTD = "minstd"
"""Identifier for minimum standard deviation attribute."""
PARAMS = "params"
"""Identifier for model parameters."""
SOCOEF = "socoef"
"""Identifier for scale/offset coefficients."""
TPCHANGE="tpchange"
"""Identifier for temporal change (TPC) metric."""

# --- Satellite States ---
DEFAULT = "default"
"""Default operational state."""
ECL = "ecl"
"""Eclipse operational state."""
MNVR = "mnvr"
"""Maneuver operational state."""
CHARGE = "charge"
"""Battery charge operational state."""
MDUMP = "mdump"
"""Momentum dump operational state."""
EUV = "euv"
"""Extreme Ultraviolet operational state."""
DISJOINT = "disjoint"
"""State indicating disjoint data or model."""
HYBRID = "hybrid"
"""Algorithm type for hybrid (multi-state) models."""

# --- Scale/Offset Indices ---
OFFSET = 0
"""Index for offset value in scale/offset pair."""
SCALE = 1
"""Index for scale value in scale/offset pair."""
SCALEOFFSET = ["offset", "scale"]
"""List of scale/offset identifiers."""

# --- Normalization Types ---
MINMAX = "minmax"
"""Normalization type: Min-Max scaling."""

# --- Boolean String Representations ---
TRUE = "true"
"""String representation for boolean True."""
FALSE = "false"
"""String representation for boolean False."""

# --- Trending Output Types ---
TRENDINGOUTPUT = [
    NONE,
    TREND,
    WLIMIT,
    ELIMIT,
    WELIMIT,
    STDDEV
]
"""List of possible trending output types."""

# --- Training Types ---
SHORTTERM = 0
"""Identifier for short-term training sessions."""
LONGTERM = 1
"""Identifier for long-term training sessions."""
TRAININGTYPES = ["ShortTerm", "LongTerm"]
"""List of training type names."""

# --- Post-training Status ---
NORMALSTATUS = 1
"""Post-training status: Normal."""
WARNINGSTATUS = 2
"""Post-training status: Warning."""
ERRORSTATUS = 3
"""Post-training status: Error."""
TRAININGERROR = 4
"""Post-training status: Training Error."""
MONITORERROR = 5
"""Post-training status: Monitor Error."""
STATUSSTRING = [
    "Unknown",
    "normal",
    "warning",
    "error",
    "training error",
    "monitor error"
]
"""List of human-readable status strings."""

INITTRAINSAMPLE = 240
"""Initial training sample size."""

# --- Algorithm Types ---
LINEAR = 1
"""Algorithm type: Linear."""
NONLINEAR = 2
"""Algorithm type: Non-linear."""
STATIC = 3
"""Algorithm type: Static (constant)."""

# --- Time Constants ---
EARTHYEAR_IN_SECONDS = 3.1556925993600003E7
"""Number of seconds in an Earth year."""
DAY_IN_SECONDS = 86400
"""Number of seconds in a day."""
HOUR_IN_SECONDS = 3600
"""Number of seconds in an hour."""

# the sample number used in training.
RETRAINSAMPLE = 2400
"""Sample size used for retraining."""

## Algorithm Names
FBNN = "fbnn"
"""Algorithm name: Feed-forward Back-propagation Neural Network."""
POLYTREND = "polytrend"
"""Algorithm name: Polynomial Trend."""
STNET = "stnet"
"""Algorithm name: State-based Neural Network (or similar)."""

#Clustering Constants:
PART_OF_CLUSTER = 0
"""Cluster point status: Part of a cluster."""
NOISE = 1
"""Cluster point status: Noise."""
UN_DETERMINED = 2
"""Cluster point status: Undetermined."""

CLUSTER = 0
"""Cluster type: Generic cluster."""
TPC_CLUSTER = 3
"""Cluster type: Temporal Change (TPC) cluster."""

def get_stat_index(stat_string: str) -> int:
    """
    Retrieves the integer index for a given statistical metric string.

    Args:
        stat_string (str): The string representation of the statistic 
            (e.g., 'max', 'min', 'mean', 'sigma').

    Returns:
        int: The corresponding integer index, or -1 if the string is undefined.
    """
    stat_map = {
        MAX: MAX_INDEX,
        MIN: MIN_INDEX,
        MEAN: MEAN_INDEX,
        SIGMA: SIGMA_INDEX
    }
    index = stat_map.get(stat_string, -1)
    if index == -1:
        logging.warning(f"Undefined statistical String: {stat_string}")
    return index
