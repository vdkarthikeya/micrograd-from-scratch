# micrograd-from-scratch

A scalar-valued autograd engine and neural network library built from scratch, following Karpathy's micrograd. Implements backpropagation over a dynamically built computation graph, then builds a fully connected MLP on top of it.

---

## What this is

This repo implements the core machinery behind neural network training — forward passes, loss computation, and backpropagation — using only Python. No PyTorch, no NumPy. Every gradient is computed manually via the chain rule.

---

## The `Value` class

The foundation is a `Value` object that wraps a scalar and tracks how it was produced. Every arithmetic operation (`+`, `*`, `**`, `tanh`, `relu`) records the child nodes that produced it and stores a `_backward` function — the local derivative of that operation.

```python
a = Value(2.0)
b = Value(3.0)
c = a * b  # c._backward knows how to push gradients back to a and b
```

When you call `.backward()` on the output, it topologically sorts the computation graph and calls `_backward` at each node, accumulating gradients via the chain rule:

```
grad_a = grad_c * (local derivative of c with respect to a)
```

This is backpropagation — not a layer-level concept, but a node-level one. Layers are an abstraction built on top of this graph.

---

## Neural network structure

### Neuron
A single neuron holds a list of weights and a bias. Given inputs `x`, it computes:

```
output = activation(w · x + b)
```

The activation function (tanh, relu, etc.) applies a nonlinearity to the weighted sum. Without it, stacking layers would just be repeated linear transformations — the whole network would collapse to a single linear function regardless of depth.

### Layer
A layer is a list of neurons, each receiving the same inputs and producing one output value. The outputs of one layer become the inputs to the next.

### MLP
An MLP chains multiple layers. Given an input vector, data flows left to right through each layer. Loss is computed at the output. Gradients flow right to left through the computation graph.

---

## Training loop

```python
for step in range(epochs):
    # Forward pass
    preds = [model(x) for x in X]

    # Loss (MSE)
    loss = sum((p - y)**2 for p, y in zip(preds, Y))

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Gradient descent update
    for p in model.parameters():
        p.data -= lr * p.grad
```

Backprop computes `p.grad` for every weight and bias in the network. The update rule `w = w - lr * w.grad` is gradient descent. The learning rate controls step size — too large and training diverges, too small and it crawls.

This loop repeats until loss decreases and predictions converge toward true values.

---

## Files

| File | Description |
|---|---|
| `micrograd.ipynb` | `Value` class, autograd engine, MLP implementation, and annotated backprop derivations in comments |
| `micrograd.pdf` | Hand-derived forward and backward pass for a 2-layer net — full chain rule derivation of ∂L/∂w₁, ∂L/∂w₂, ∂L/∂b₁, ∂L/∂b₂ |

---

## Key insight

Backpropagation is just the chain rule applied systematically across a computation graph, from output node back to every weight. Each node only needs to know its local derivative and the gradient flowing in from the right. The rest is bookkeeping.
