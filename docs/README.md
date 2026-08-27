# spork-pds Documentation

`spork-pds` is a standalone CPython extension exposing immutable persistent collections and their transient batch-update forms.

## Documents

### [API Reference](API.md)

Constructors, exported types and constants, persistent operations, transient operations, Python protocol support, and typed-vector interoperability.

### [Design and Complexity](DESIGN.md)

The implementation model, structural sharing, transient ownership, collection internals, and expected operation costs.

### [Benchmarks](BENCHMARKS.md)

How to run and extend the benchmark suite that compares `spork-pds` with Python's built-in mutable collections.

## Start here

```python
from spork_pds import hash_map, hash_set, vec

vector = vec(1, 2, 3)
map_value = hash_map("answer", 42)
set_value = hash_set([1, 2, 3])
```

See the project [README](../README.md) for installation and development instructions.
