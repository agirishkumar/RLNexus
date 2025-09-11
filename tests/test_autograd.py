import numpy as np
from rlnexus.linalg.core import Tensor

def test_autograd_add():
    """
    Tests the backward pass for the addition operation.
    """
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    z = x + y

    z.backward()

    np.testing.assert_array_equal(x.grad, np.array([1, 1, 1]))
    np.testing.assert_array_equal(y.grad, np.array([1, 1, 1]))

def test_autograd_mul():
    """
    Tests the backward pass for the multiplication operation.
    """
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    z = x * y # z = [4, 10, 18]

    z.backward()

    # dz/dx = y
    np.testing.assert_array_equal(x.grad, y.data)
    # dz/dy = x
    np.testing.assert_array_equal(y.grad, x.data)

def test_autograd_chain_rule():
    """
    Tests a more complex graph to verify the chain rule.
    """
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = Tensor(4.0)

    # a = x * y  (a=6)
    # b = a + z  (b=10)
    a = x * y
    b = a + z

    b.backward()

    # db/dx = y = 3
    # db/dy = x = 2
    # db/dz = 1
    # db/da = 1
    np.testing.assert_allclose(x.grad, 3.0)
    np.testing.assert_allclose(y.grad, 2.0)
    np.testing.assert_allclose(z.grad, 1.0)
    np.testing.assert_allclose(a.grad, 1.0)

def test_autograd_pow():
    """
    Tests the backward pass for the power operation.
    """
    x = Tensor([2.0, 3.0, 4.0])
    y = x**3 # y = [8, 27, 64]

    y.backward()

    # dy/dx = 3 * x**2 = [12, 27, 48]
    np.testing.assert_allclose(x.grad, [12.0, 27.0, 48.0])

def test_autograd_composed_ops():
    """
    Tests backward pass for operations composed of others (sub, div).
    """
    a = Tensor(10.0)
    b = Tensor(5.0)

    # Test subtraction: c = a - b
    c = a - b
    c.backward()
    np.testing.assert_allclose(a.grad, 1.0)
    np.testing.assert_allclose(b.grad, -1.0)

    # Reset grads for next test
    a.grad = None
    b.grad = None

    # Test division: d = a / b
    d = a / b
    d.backward()
    # d/da = 1/b = 1/5 = 0.2
    # d/db = -a/b^2 = -10/25 = -0.4
    np.testing.assert_allclose(a.grad, 0.2)
    np.testing.assert_allclose(b.grad, -0.4)

def test_autograd_dot():
    """
    Tests the backward pass for matrix multiplication.
    """
    # Test with vectors
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a.dot(b) # c = 1*4 + 2*5 + 3*6 = 32
    c.backward()
    np.testing.assert_allclose(a.grad, b.data)
    np.testing.assert_allclose(b.grad, a.data)

    # Test with matrices
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    # c = [[19, 22], [43, 50]]
    c = a.dot(b)
    c.backward()

    # dc/da = c.grad @ b.T
    # dc/db = a.T @ c.grad
    # c.grad is ones
    np.testing.assert_allclose(a.grad, [[11, 15], [11, 15]])
    np.testing.assert_allclose(b.grad, [[4, 4], [6, 6]])

def test_autograd_transpose():
    """
    Tests the backward pass for transpose operation.
    """
    x = Tensor([[1, 2, 3], [4, 5, 6]]) # shape (2, 3)
    y = x.T # shape (3, 2)
    
    # To make y's grad not just ones, let's sum it
    z = y.sum()
    z.backward()
    
    # z = x[0,0] + x[1,0] + ...
    # dz/dx_ij = 1 for all i,j
    np.testing.assert_allclose(x.grad, np.ones_like(x.data))

def test_autograd_getitem():
    """
    Tests the backward pass for the getitem operation.
    """
    x = Tensor([10, 20, 30])
    y = x[1] # y = 20
    y.backward()

    # dy/dx = [0, 1, 0]
    np.testing.assert_allclose(x.grad, [0, 1, 0])

    # Test with a slice
    x = Tensor([10, 20, 30, 40])
    z = x[1:3] # z = [20, 30]
    z = z.sum()
    z.backward()
    np.testing.assert_allclose(x.grad, [0, 1, 1, 0])

def test_autograd_broadcasting():
    """
    Tests the backward pass for broadcasting operations (scalar * vector).
    """
    # Test scalar * vector
    scalar = Tensor(2.0)
    vector = Tensor([1.0, 2.0, 3.0])
    result = scalar * vector  # [2.0, 4.0, 6.0]
    
    loss = result.sum()  # scalar
    loss.backward()
    
    # d(loss)/d(scalar) = sum(vector) = 6.0
    # d(loss)/d(vector) = scalar = [2.0, 2.0, 2.0]
    np.testing.assert_allclose(scalar.grad, 6.0)
    np.testing.assert_allclose(vector.grad, [2.0, 2.0, 2.0])
