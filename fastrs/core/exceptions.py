class FastrsError(Exception):
    """Base class for exceptions raised by Fastrs."""

    pass


class PreprocessingError(FastrsError):
    """Exception raised for errors related to preprocessing."""

    pass


class TrainingError(FastrsError):
    """Exception raised for errors related to training."""

    pass


class UtilError(FastrsError):
    """Exception raised for errors related to utility functions."""

    pass


class ReducerError(FastrsError):
    """Exception raised for errors related to dimensionality reduction."""

    pass


class ItemError(FastrsError):
    """Exception raised for errors related to Item processing."""

    pass
