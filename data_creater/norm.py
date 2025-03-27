#  normalization operator for model training
import numpy as np


# l2_norm for np.ndarray
# imply l2_normalize to several dimensions to the input
def l2_norm(input_array, axes):
    """
    Compute the L2 norm along the specified axes of the input array and normalize the array.

    Parameters:
    input_array (np.ndarray): The input multi-dimensional array.
    axes (int or tuple of ints): The axes along which to compute the L2 norm and normalize.
                                 If an integer is provided, it will be treated as a single axis.
                                 If a tuple is provided, the norm will be computed along those axes.

    Returns:
    np.ndarray: The normalized array with L2 norm equal to 1 along the specified axes.
    """
    # Ensure that axes is a tuple, even if only a single axis is provided
    if isinstance(axes, int):
        axes = (axes,)

    # Compute the F-norm along the specified axes
    norm = np.linalg.norm(input_array, ord='fro', axis=axes, keepdims=True)

    # Normalize the array, ensuring no division by zero
    normalized_array = input_array / np.maximum(norm, np.finfo(input_array.dtype).eps)

    return normalized_array
