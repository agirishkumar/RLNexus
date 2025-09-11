import numpy as np
from rlnexus.linalg.core import Tensor

def test_tensor_initialization():
    """
    Tests that the Tensor class correctly initializes with various data types.
    """
    # Test with a list
    data_list = [[1, 2], [3, 4]]
    tensor = Tensor(data_list)
    assert isinstance(tensor.data, np.ndarray)
    np.testing.assert_array_equal(tensor.data, np.array(data_list))

    # Test with a NumPy array
    data_np = np.array([5, 6, 7])
    tensor_np = Tensor(data_np)
    assert isinstance(tensor_np.data, np.ndarray)
    np.testing.assert_array_equal(tensor_np.data, data_np)
