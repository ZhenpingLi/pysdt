import os
import sys
from typing import List, Optional

from config.sdt_constants import DISJOINT
from sdtdb.sdt_db import AlgorithmType, StateType

# Add parent directory to path to find sdtdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Constants
DERIVATIVETYPE = 1
MIXEDTYPE = 2
NORMALTYPE = 0
INFO = "INFO"
SHORTTERM = 0
DR = "dr"
INDR = "indr"
MIXED = "mixed"
NORMALIZE = "norm"
RETRAIN = "retrain"
MONITOR = "monitor"


class AlgorithmDef:
    """
    Represents the operational definition of a machine learning algorithm.
    
    This class parses and encapsulates the properties, options, and 
    configurations defined in the AIMS database (via AlgorithmType). It acts 
    as a high-level configuration object for training workers and trends.
    """

    def __init__(self, al_type: AlgorithmType):
        """
        Initializes a new instance of AlgorithmDef.

        Args:
            al_type (AlgorithmType): The database object containing the algorithm definition.
        """
        self.al_type = al_type
        self.is_retraining = False
        self.is_normalized = False
        self.is_monitoring = False
        self.is_derivative = False
        self.checkdqf_flag = False
        self.input_processing_type = NORMALTYPE
        self.dim: List[int] = []
        self.np : int = 0
        
        self.init()

    def get_name(self) -> str:
        """Returns the name of the algorithm (e.g., 'fbnn', 'ridgefe')."""
        return self.al_type.name

    def init(self):
        """
        Parses the 'option', 'dim', 'np', and 'attribute' fields from the 
        underlying AlgorithmType to set internal configuration flags.
        """
        if not self.al_type:
            return
        self.is_retraining = False
        self.is_normalized = False
        self.is_monitoring = False
        self.is_derivative = False
        self.checkdqf_flag = False
        self.input_processing_type = NORMALTYPE
        
        properties = self.al_type.option
        if properties:
            tokens = properties.split("|")
            for token in tokens:
                if token == RETRAIN:
                    self.is_retraining = True
                elif token == NORMALIZE:
                    self.is_normalized = True
                elif token == MONITOR:
                    self.is_monitoring = True
                elif token == DR:
                    self.is_derivative = True
                elif token == INDR:
                    self.input_processing_type = DERIVATIVETYPE
                elif token == MIXED:
                    self.input_processing_type = MIXEDTYPE
        
        if self.al_type.dim:
            self.dim = self.al_type.dim
            
        self.np = int(self.al_type.np) if self.al_type.np is not None else 0

        checkdqf_s = self.get_attribute("checkdqf")
        self.checkdqf_flag = checkdqf_s is not None

    def is_normalized_check(self) -> bool:
        """Returns True if the algorithm requires input data normalization."""
        return self.is_normalized

    def set_normalized(self, normalized: bool):
        """Sets the normalization flag."""
        self.is_normalized = normalized

    def is_retraining_check(self) -> bool:
        """Returns True if the algorithm supports/requires retraining cycles."""
        return self.is_retraining

    def is_monitoring_check(self) -> bool:
        """Returns True if monitoring mode is enabled for this algorithm."""
        return self.is_monitoring

    def is_derivative_check(self) -> bool:
        """Returns True if the algorithm is trained on derivative data."""
        return self.is_derivative

    def get_input_processing_type(self) -> int:
        """Returns the type of input processing required (NORMAL, DERIVATIVE, MIXED)."""
        return self.input_processing_type

    def check_dqf(self) -> bool:
        """Returns True if Data Quality Flag (DQF) filtering is required."""
        return self.checkdqf_flag

    def set_check_dqf(self, check: bool):
        """Enables the DQF filtering flag."""
        self.checkdqf_flag = True

    def get_dimension(self) -> List[int]:
        """Returns the dimensionality parameters (e.g., hidden layers, polynomial degree)."""
        return self.dim

    def get_np(self) -> int:
        """Returns the number of pattern cycles included in the training."""
        return self.np

    def get_states(self) -> Optional[List[StateType]]:
        """Returns the list of operational states defined for this algorithm."""
        return self.al_type.state if self.al_type.state else None

    def get_max_diff(self) -> float:
        """
        Retrieves the 'maxdiff' attribute value used for outlier thresholds.

        Returns:
            float: The maximum difference threshold, or 0.0 if not defined.
        """
        maxdiff_string = self.get_attribute("maxdiff")
        if maxdiff_string:
            return float(maxdiff_string)
        else:
            return 0.0

    def get_attribute(self, attr_name: str) -> Optional[str]:
        """
        Searches the 'attribute' list for a specific key-value pair.
        Attributes are expected in 'key|value' format.

        Args:
            attr_name (str): The name of the attribute to retrieve.

        Returns:
            Optional[str]: The attribute value as a string, or None if not found.
        """
        if self.al_type.attribute:
            for attr_dict in self.al_type.attribute:
                attrs = attr_dict.split('|')
                if len(attrs) >= 2 and attrs[0] == attr_name:
                    return "|".join(attrs[1:])
        return None

    def get_so_coef(self) -> Optional[List[float]]:
        """
        Retrieves the scale and offset coefficients from the 'socoef' attribute.

        Returns:
            Optional[List[float]]: A list [scale, offset], or None if not found.
        """
        socoef_string = self.get_attribute("socoef")
        if socoef_string:
            tokens = socoef_string.split("|")
            if len(tokens) >= 2:
                return [float(tokens[0]), float(tokens[1])]
        return None

    def get_attribute_double(self, attr_name: str) -> float:
        """
        Retrieves an attribute and converts it to a float.

        Args:
            attr_name (str): The name of the attribute.

        Returns:
            float: The numeric value, or 0.0 if missing or invalid.
        """
        attr_value = self.get_attribute(attr_name)
        if attr_value:
            try:
                return float(attr_value)
            except ValueError:
                return 0.0
        else:
            return 0.0

    def get_state(self, state: str) -> Optional[StateType]:
        """
        Finds a specific state definition by name.

        Args:
            state (str): The name of the state (e.g., 'ECL').

        Returns:
            Optional[StateType]: The state object, or None if not found.
        """
        state_list = self.al_type.state
        if state_list is None:
            return None
        else:
            for type_obj in state_list:
                if state == type_obj.name:
                    return type_obj
            return None

    def is_disjoint(self, state : str) -> bool:
        """
        Checks if a specific state is marked as disjoint in its flag property.

        Args:
            state (str): The name of the state.

        Returns:
            bool: True if the state is disjoint, False otherwise.
        """
        state_type = self.get_state(state)
        if state_type is None:
            return False
        else:
            return state_type.flag == DISJOINT
