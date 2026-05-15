from enum import Enum
import numpy as np

class DataType(Enum):
    """
    Docstring for DataType

    Type of data as name and min_value and max_value 
    in value[0] and value[1], respectively.
    """
    ANGLES = (0, 360)
    RADS = (0, 2*np.pi)
    HOURS = (0, 24)
    MONTHS = (0, 12)
    PERCENT = (0, 1)