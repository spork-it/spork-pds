# spork-pds

[![Tests](https://github.com/spork-it/spork-pds/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/spork-it/spork-pds/actions/workflows/test.yml)
![PyPI - Version](https://img.shields.io/pypi/v/spork-pds)

`spork-pds` provides fast, immutable persistent data structures for CPython. They use familiar collection operators, but return new values while sharing unchanged structure with the original. Snapshots stay inexpensive and previous values remain unchanged.

The package is the standalone home of the persistent data structures originally developed for [Spork](https://github.com/spork-it/spork-lang). It has no dependency on the Spork language or runtime.

> **Alpha:** The API and binary compatibility may change between releases.

## Features

- `Vector`: bit-partitioned persistent vector with indexing, slicing, `+` concatenation, and `*` repetition
- `Map`: persistent hash map backed by a HAMT with dict-style `|` merging
- `Set`: persistent hash set with `|`, `&`, `-`, and `^` operations
- `SortedVector`: ordered persistent collection backed by a size-annotated red-black tree
- `Cons`: immutable linked-list cells
- `DoubleVector` and `IntVector`: specialized float64 and int64 vectors with the read-only buffer protocol
- Transient variants for efficient batches of controlled mutation
- Structural ABC integration, hashing, iteration, generic aliases, and pickle support
- CPython 3.10+ support, including free-threaded CPython 3.14 builds

## Installation

```bash
python -m pip install spork-pds
```

A C compiler and Python development headers are required when installing from a source distribution. Published releases are intended to provide wheels for supported Python versions and platforms.

## Quick start

```python
from spork.pds import Map, Set, Vector, sorted_vec

numbers = Vector([1, 2, 3])
extended = numbers + [4, 5]
repeated = numbers * 2

assert list(numbers) == [1, 2, 3]
assert list(extended) == [1, 2, 3, 4, 5]
assert list(repeated) == [1, 2, 3, 1, 2, 3]

config = Map({"host": "localhost", "port": 8000})
production = config | {"host": "example.com"}

assert config["host"] == "localhost"
assert production["host"] == "example.com"

roles = Set(["reader", "writer"])
admin_roles = roles | {"admin"}

assert "admin" not in roles
assert "admin" in admin_roles

ordered = sorted_vec([5, 1, 3, 2, 4])
assert list(ordered) == [1, 2, 3, 4, 5]
```

### Native operators, persistent values

Operators always produce persistent `spork.pds` collections and leave their operands unchanged:

```python
updated_map = config | {"port": 443}
combined_set = roles | {"admin", "auditor"}
reduced_set = combined_set - {"reader"}
longer_vector = numbers + range(4, 7)
```

Augmented assignment follows Python's normal immutable-value behavior. It rebinds the name rather than mutating the collection:

```python
original = Map({"users": 100})
updated = original
updated |= {"users": 101}

assert original["users"] == 100
assert updated["users"] == 101
```

The named persistent operations—such as `.assoc()`, `.conj()`, and `.disj()`—remain available when an individual update is clearer.

### Batch updates with transients

Persistent updates are ideal when each intermediate version matters. For a batch where only the final value matters, use a transient:

```python
from spork.pds import EMPTY_VECTOR

builder = EMPTY_VECTOR.transient()
for value in range(100_000):
    builder.conj_mut(value)

values = builder.persistent()
assert values[-1] == 99_999
```

Calling `persistent()` invalidates the transient. Further edits and element access raise `RuntimeError`; discard the transient immediately after conversion.

### Typed vectors and NumPy

```python
from spork.pds import vec_f64, vec_i64

floats = vec_f64(1.0, 2.0, 3.0)
integers = vec_i64(1, 2, 3)

# The exported buffers are read-only.
assert memoryview(floats).format == "d"
assert memoryview(integers).format == "q"

# NumPy can view the vectors through the buffer protocol.
import numpy as np
array = np.asarray(floats)
```

The first buffer request materializes and caches contiguous storage; subsequent views reuse that immutable cache.

## Documentation

- [Practical guide](docs/GUIDE.md)
- [API reference](docs/API.md)
- [Design and complexity](docs/DESIGN.md)
- [Benchmark suite](docs/BENCHMARKS.md)
- [Documentation index](docs/README.md)
- [Changelog](CHANGELOG.md)

## Development

Clone the repository and set up the development environment:

```bash
git clone https://github.com/spork-it/spork-pds.git
cd spork-pds

make venv
make test
make fuzz
```

Useful targets:

```bash
make build
make build-debug
make benchmark BENCH_ARGS="--size 100000 --iter 50"
make dist
make check-dist
```

See `make help` for the complete target list.

## License

MIT. See [LICENSE](LICENSE).
