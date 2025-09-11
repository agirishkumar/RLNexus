# The Autograd Engine

The heart of RLNexus's deep learning capabilities is a custom automatic differentiation (autograd) engine, built from the ground up using only NumPy.

## The `Tensor` Class

The central component of the engine is the `rlnexus.linalg.core.Tensor` class. This class is a wrapper around a `numpy.ndarray` that keeps track of the operations performed on it, allowing for the automatic computation of gradients.

### Computation Graph

Every time an operation is performed on `Tensor` objects, a new `Tensor` is created that stores the result. This new tensor also maintains a reference to its parent tensors and the operation that created it. This forms a dynamic computation graph.

For example, when you execute `c = a + b`, where `a` and `b` are `Tensor`s, the resulting `Tensor` `c` knows that it was created by an addition operation with parents `a` and `b`.

### Backpropagation

When you call the `.backward()` method on a final tensor (typically a scalar loss value), the autograd engine performs two key steps:

1.  **Topological Sort**: It traverses the computation graph from the final node backwards to create a linear ordering of all participating tensors.
2.  **Chain Rule Application**: It proceeds through the sorted list in reverse, and for each tensor, it calls its local `_backward()` function. This function computes the gradient with respect to its parents and adds it to their `.grad` attributes.

This process efficiently applies the chain rule across the entire graph, populating the `.grad` attribute of every leaf tensor that requires a gradient.

Our engine supports a variety of operations, including:
- All basic arithmetic (`+`, `-`, `*`, `/`, `**`)
- Matrix multiplication (`.dot()`)
- Indexing (`__getitem__`)
- Transposition (`.T`)
- Activation functions (`.relu()`)
- Reduction ops (`.sum()`, `.mean()`)
