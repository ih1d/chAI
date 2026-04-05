# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Chai is a C implementation of Karpathy's micrograd — a scalar-valued autograd engine with backpropagation over a dynamically built computation graph.

## Build Commands

```bash
# Compile and run the binary
cc -o chai src/main.c src/value.c -lm
./chai

# Build as a static library
make            # produces libchai.a
make clean      # remove build artifacts
```

## Architecture

The entire autograd engine lives in a single module:

- **`include/value.h`** — `Value` struct (scalar data, grad, children, op, backward fn pointer), `Op` enum, and all operation declarations
- **`src/value.c`** — Implementation: forward ops (add, mul, pow, relu, neg, sub, truediv), per-op backward functions, topological sort for backprop, and memory management
- **`src/main.c`** — Test/demo entry point

Each operation allocates a new `Value`, stores its inputs as `children`, and attaches a `_backward` function pointer. `backward()` builds a topological ordering via DFS, then calls `_backward` in reverse order to propagate gradients.

The `pow_val` exponent is stored as a phantom child `Value` (children[1]) so `_backward` can access it without changing the struct.
