# spork-pds Documentation

`spork-pds` is a standalone CPython extension providing immutable persistent collections and single-use transient builders.

> **Version note:** These documents describe the current `main` branch and may be ahead of the latest PyPI release. The project is alpha and APIs may change between releases.

## Choose a document

| If you want to… | Start with… |
| --- | --- |
| Choose a collection or follow common usage patterns | [Practical Guide](GUIDE.md) |
| Look up constructors, operators, methods, or Python protocols | [API Reference](API.md) |
| Understand structural sharing and operation costs | [Design and Complexity](DESIGN.md) |
| Run, interpret, or extend the benchmark suite | [Benchmarks](BENCHMARKS.md) |
| Review release changes | [Changelog](../CHANGELOG.md) |

## Quick example

```python
from spork_pds import Map, Set, Vector

base = Vector([1, 2, 3])
updated = base.conj(4)

assert list(base) == [1, 2, 3]
assert list(updated) == [1, 2, 3, 4]

config = Map({"host": "localhost"}) | {"port": 8000}
tags = Set(["alpha"]) | {"documented"}
```

Persistent operations return new values and leave existing values unchanged. Internally, versions share structure that did not need to change.

For a large batch, use a transient and convert it once:

```python
builder = Vector().transient()
for value in range(10_000):
    builder.append(value)
result = builder.persistent()
```

The transient is invalid after `.persistent()` and should be discarded.

## Verify the examples

Run `make verify-docs` from the repository root. It executes every Python fence in a fresh environment with the public API preloaded.

## Scope

The distribution name is `spork-pds`; the Python import is `spork_pds`. It has no dependency on the [Spork language](https://github.com/spork-it/spork-lang).

See the project [README](../README.md) for installation, supported Python versions, development setup, and release-level highlights.
