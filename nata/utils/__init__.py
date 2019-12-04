import numpy as np


def props_as_arr(instances, property_, array_type, sorted: bool = True):
    """Takes instances and returns its properties as array."""
    if sorted:
        return np.sort(
            np.array(
                [getattr(obj, property_) for obj in instances], dtype=array_type
            )
        )
    else:
        return np.array(
            [getattr(obj, property_) for obj in instances], dtype=array_type
        )
