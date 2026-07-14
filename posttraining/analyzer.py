import os
import sys
from abc import ABC, abstractmethod
from typing import List

from algorithm.subsystem_output import SubsystemOutput
from posttraining.mnemonic_status import MnemonicStatus

# Add parent directory to path to find other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Analyzer(ABC):
    """
    Abstract Base Class (ABC) defining the interface for post-training analysis.
    
    Implementations of this interface are responsible for parsing the complex outputs 
    of training or real-time monitoring sessions (e.g., model residuals, standard 
    deviations, outlier clusters) to determine the overall operational health 
    and safety status of satellite subsystems.
    """
    
    # Analysis Mode Constants
    TRAININGOUTPUT = 0
    """Mode for analyzing the direct output of a training session."""
    
    RTMONITOR = 1
    """Mode for analyzing results from a real-time monitoring process."""

    @abstractmethod
    def analyze(self, data_list: List[SubsystemOutput]) -> List[MnemonicStatus]:
        """
        Evaluates the results of a completed data training session.

        This method analyzes the trained models and their statistics to generate 
        an initial health baseline for all mnemonics in the provided subsystems.

        Args:
            data_list (List[SubsystemOutput]): The aggregated training results 
                grouped by subsystem.

        Returns:
            List[MnemonicStatus]: A list of health status records for every 
                successfully analyzed mnemonic.
        """
        pass

    @abstractmethod
    def get_status(self, data_list: List[SubsystemOutput], start: float, end: float) -> List[MnemonicStatus]:
        """
        Determines the operational status of subsystems within a specific time range.
        
        Typically used in monitoring mode to assess the current state of 
        telemetry relative to the trained models.

        Args:
            data_list (List[SubsystemOutput]): The output data from the monitor.
            start (float): The start timestamp of the evaluation period.
            end (float): The end timestamp of the evaluation period.

        Returns:
            List[MnemonicStatus]: A list of status records defining the 
                health level (normal, warning, error) for each subsystem.
        """
        pass
