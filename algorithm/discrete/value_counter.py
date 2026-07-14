from typing import List


class ValueCounter:
    """
    Utility for counting and averaging occurrences of values within a tolerance.
    
    This class groups similar numeric values together. When a new value 
    falls within the specified 'bound' of the group's current representative 
    value, it is added to the group, and the representative value is 
    recalculated as the running average of all group members.
    """

    def __init__(self, value: float, bound: float):
        """
        Initializes a new ValueCounter.

        Args:
            value (float): The initial value for the group.
            bound (float): The grouping tolerance. If 0, only exact matches 
                are grouped.
        """
        self.value: float = value
        self.number: int = 1
        self.bound: float = bound

    def match(self, v: float) -> bool:
        """
        Attempts to add a new value to this group.
        
        If the value falls within the tolerance, it is integrated and the 
        group's average value is updated.

        Args:
            v (float): The value to check.

        Returns:
            bool: True if the value was added to this group, False otherwise.
        """
        is_match = False
        if self.bound > 0:
            if abs(self.value - v) <= self.bound:
                is_match = True
        else: # Exact match required
            if self.value == v:
                is_match = True
        
        if is_match:
            # Update the representative value to be the new average
            self.value = (self.value * self.number + v) / (self.number + 1)
            self.number += 1
            return True
        
        return False

    def get_number(self) -> int:
        """Returns the total number of members in this group."""
        return self.number

    def get_value(self) -> float:
        """Returns the current average value representing this group."""
        return self.value

    def __repr__(self) -> str:
        """Returns a string representation of the counter."""
        return f"ValueCounter(value={self.value:.4f}, count={self.number}, bound={self.bound})"


def get_value_counts(data: List[float], bound: float) -> List[ValueCounter]:
    """
    Groups a sequence of numbers into a set of ValueCounter objects.
    
    This acts as a simple 1D clustering algorithm to identify the 
    predominant discrete levels in a dataset.

    Args:
        data (List[float]): The sequence of numeric values.
        bound (float): The grouping tolerance.

    Returns:
        List[ValueCounter]: A list of objects representing the identified groups.
    """
    if not data:
        return []

    counters: List[ValueCounter] = []
    
    for value in data:
        found_match = False
        # Try to integrate into an existing group
        for counter in counters:
            if counter.match(value):
                found_match = True
                break
        
        # Start a new group if no existing group is close enough
        if not found_match:
            counters.append(ValueCounter(value, bound))
            
    return counters
