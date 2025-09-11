import numpy as np
from rlnexus.linalg.nn import Linear
from rlnexus.linalg.core import Tensor

def test_linear_layer():
    """
    Tests the forward and backward pass of the Linear layer.
    """
    batch_size, in_features, out_features = 4, 10, 5
    
    layer = Linear(in_features, out_features)
    x = Tensor(np.random.randn(batch_size, in_features))
    
    # Forward pass
    y = layer(x)
    assert y.data.shape == (batch_size, out_features)
    
    # Create a dummy gradient for the output
    y_grad = np.random.randn(batch_size, out_features)
    y.grad = y_grad
    
    # Backward pass
    y.backward()

    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    
    assert layer.weight.grad.shape == (out_features, in_features)
    assert layer.bias.grad.shape == (out_features,)
    assert x.grad.shape == (batch_size, in_features)

from rlnexus.linalg.nn import ReLU

def test_relu():
    """
    Tests the forward and backward pass of the ReLU activation.
    """
    relu = ReLU()
    x = Tensor([-1, 0, 1, 2])
    
    # Forward pass
    y = relu(x)
    np.testing.assert_allclose(y.data, [0, 0, 1, 2])
    
    # Backward pass
    y.backward()
    
    # Grad is 1 where x > 0, else 0
    np.testing.assert_allclose(x.grad, [0, 0, 1, 1])

from rlnexus.linalg.nn import Sequential

def test_sequential():
    """
    Tests the Sequential module.
    """
    in_features, h1_features, out_features = 10, 20, 5
    
    model = Sequential(
        Linear(in_features, h1_features),
        ReLU(),
        Linear(h1_features, out_features)
    )
    
    # Check parameters
    params = model.parameters()
    assert len(params) == 4 # 2 weights, 2 biases
    assert params[0].data.shape == (h1_features, in_features)
    assert params[1].data.shape == (h1_features,)
    assert params[2].data.shape == (out_features, h1_features)
    assert params[3].data.shape == (out_features,)
    
    # Check forward pass
    batch_size = 4
    x = Tensor(np.random.randn(batch_size, in_features))
    y = model(x)
    assert y.data.shape == (batch_size, out_features)
