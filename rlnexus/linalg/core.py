import numpy as np

class Tensor:
    """
    A wrapper for NumPy arrays to support automatic differentiation.
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = None

        # Internal variables for autograd graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the operation that produced this tensor

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)

            # Handle broadcasting for gradients
            def unbroadcast_add(grad, target_shape):
                """Unbroadcast gradient for addition operation."""
                if grad.shape == target_shape:
                    return grad

                # If target is scalar, sum all elements
                if target_shape == () or target_shape == (1,):
                    return np.array(np.sum(grad))

                # Handle common broadcasting cases
                if len(grad.shape) == len(target_shape):
                    # Same number of dimensions, check if we need to sum along certain axes
                    result = grad.copy()
                    for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
                        if target_dim == 1 and grad_dim > 1:
                            result = np.sum(result, axis=i, keepdims=True)
                    return result.reshape(target_shape)
                else:
                    # Different number of dimensions - sum over extra dimensions
                    result = grad
                    while len(result.shape) > len(target_shape):
                        result = np.sum(result, axis=0)

                    # If still not matching, try to reshape
                    if result.shape != target_shape:
                        # Try summing along last dimension if needed
                        if len(result.shape) == len(target_shape) + 1:
                            result = np.sum(result, axis=-1)

                    return result.reshape(target_shape) if result.shape != target_shape else result

            self.grad += unbroadcast_add(out.grad, self.data.shape)
            other.grad += unbroadcast_add(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)

            # Compute gradients
            grad_wrt_self = other.data * out.grad
            grad_wrt_other = self.data * out.grad
            
            # Handle broadcasting by summing over broadcasted dimensions
            # This handles scalar*vector, scalar*matrix, etc. automatically
            def unbroadcast(grad, target_shape):
                # Sum over dimensions that were added by broadcasting
                ndims_added = grad.ndim - len(target_shape)
                if ndims_added > 0:
                    sum_axes = tuple(range(ndims_added))
                    grad = np.sum(grad, axis=sum_axes)
                
                # Sum over dimensions that were broadcasted from size 1
                for i, (dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
                    if target_dim == 1 and dim != 1:
                        grad = np.sum(grad, axis=i, keepdims=True)
                
                return grad.reshape(target_shape)

            self.grad += unbroadcast(grad_wrt_self, self.data.shape)
            other.grad += unbroadcast(grad_wrt_other, other.data.shape)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power only supported for scalars"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other**-1)

    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data.dot(other.data), (self, other), 'dot')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)
            
            if self.data.ndim == 1 and other.data.ndim == 1:
                # Vector-vector dot product
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            else:
                # Matrix multiplication
                self.grad += out.grad.dot(other.data.T)
                other.grad += self.data.T.dot(out.grad)

        out._backward = _backward

        return out

    def transpose(self, dims):
        out = Tensor(np.transpose(self.data, dims), (self,), 'transpose')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += np.transpose(out.grad, dims)
        out._backward = _backward
        
        return out

    @property
    def T(self):
        return self.transpose(dims=(1, 0))

    def __getitem__(self, val):
        out = Tensor(self.data[val], (self,), 'getitem')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            np.add.at(self.grad, val, out.grad)
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward

        return out

    def mean(self, axis=None, keepdims=False):
        out = self.sum(axis=axis, keepdims=keepdims) * (1.0 / np.prod(self.data.shape))
        return out

    def softmax(self, axis=-1):
        """
        Compute softmax along the specified axis.
        """
        # Subtract max for numerical stability
        x_max = self.data.max(axis=axis, keepdims=True)
        exp_x = (self - x_max).exp()
        sum_exp_x = exp_x.sum(axis=axis, keepdims=True)
        return exp_x / sum_exp_x

    def exp(self):
        """
        Compute element-wise exponential.
        """
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        """
        Compute element-wise natural logarithm.
        """
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Iterative topological sort of the graph (avoids recursion depth issues)
        topo = []
        visited = set()
        temp_visited = set()
        stack = [self]

        while stack:
            v = stack[-1]

            if v in visited:
                stack.pop()
                continue

            if v in temp_visited:
                # All children have been visited, add to topological order
                stack.pop()
                visited.add(v)
                topo.append(v)
                continue

            # Mark as temporarily visited
            temp_visited.add(v)

            # Add all children to stack
            for child in v._prev:
                if child not in visited and child not in temp_visited:
                    stack.append(child)

        # Initialize gradient of the final tensor if it's not already set
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # Backpropagate through the graph
        for v in reversed(topo):
            v._backward()
