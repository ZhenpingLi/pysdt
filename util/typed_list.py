from typing import Type, Any, List


class TypedList:
    """
    A list-like container that enforces type consistency for its elements.
    
    This class mimics a fixed-dimension array but ensures that every item added 
    matches the specified type. It is useful for maintaining data integrity 
    in complex data processing pipelines where specific object types are 
    expected.
    """

    def __init__(self, dimension: int, type_to_enforce: Type):
        """
        Initializes the TypedList with a fixed size and target type.

        Args:
            dimension (int): The number of elements the list should hold.
            type_to_enforce (Type): The Python class or type to enforce 
                (e.g., DataTrend, int).
        """
        self._data = [None] * dimension
        self._type = type_to_enforce

    def __setitem__(self, index: int, value: Any):
        """
        Sets an element at the specified index, enforcing type integrity.

        Args:
            index (int): The position to update.
            value (Any): The value to insert.

        Raises:
            TypeError: If the value is not an instance of the enforced type.
        """
        if value is not None and not isinstance(value, self._type):
            raise TypeError(f"TypedList: Expected {self._type}, got {type(value)}")
        self._data[index] = value

    def __getitem__(self, index: int) -> Any:
        """
        Retrieves the element at the specified index.

        Args:
            index (int): The position to retrieve.

        Returns:
            Any: The element at the index.
        """
        return self._data[index]

    @property
    def data(self) -> List[Any]:
        """
        Returns the underlying Python list.

        Returns:
            List[Any]: The raw data list.
        """
        return self._data
