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
            
            self.grad += out.grad
            other.grad += out.grad if other.data.shape == out.data.shape else np.sum(out.grad, axis=0)

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

    def backward(self):
        # Topological sort of the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Initialize gradient of the final tensor if it's not already set
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # Backpropagate through the graph
        for v in reversed(topo):
            v._backward()
