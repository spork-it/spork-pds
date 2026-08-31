# spork-pds

[![Tests](https://github.com/spork-it/spork-pds/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/spork-it/spork-pds/actions/workflows/test.yml)
![PyPI - Version](https://img.shields.io/pypi/v/spork-pds)

`spork-pds` provides fast immutable persistent collections for CPython. Updates return new values while sharing unchanged structure; prior versions remain unchanged. The package is usable directly from Python and has no dependency on the Spork language or runtime.

## Install

```bash
python -m pip install spork-pds
```

Published releases provide wheels for supported CPython versions and platforms. Source builds require a C compiler and Python development headers.

## Quick start

```python
from spork.pds import Map, Set, Vector, sorted_vec

numbers = Vector([1, 2, 3])
extended = numbers + [4]

config = Map({"host": "localhost", "port": 8000})
production = config | {"host": "example.com"}

roles = Set(["reader", "writer"])
admin_roles = roles | {"admin"}

ordered = sorted_vec([3, 1, 2, 2])

assert list(numbers) == [1, 2, 3]
assert list(extended) == [1, 2, 3, 4]
assert config["host"] == "localhost"
assert production["host"] == "example.com"
assert "admin" not in roles and "admin" in admin_roles
assert list(ordered) == [1, 2, 2, 3]
```

Use a transient when only the final result of a large update batch matters:

```python
builder = Vector().transient()
for value in range(100_000):
    builder.append(value)
result = builder.persistent()
```

Calling `.persistent()` invalidates the transient. Discard it immediately afterward.

## Collection families

- `Vector`, `Map`, and `Set` provide general persistent collections.
- `SortedVector` retains duplicates in configured order.
- `DoubleVector` and `IntVector` store unboxed values and export read-only buffers.
- `Cons` provides immutable linked-list cells.
- Transient variants support controlled single-owner mutation batches.

On free-threaded CPython 3.14t, persistent values may be shared across threads and independent transient builders may execute in parallel. Each transient is confined to its creating thread. Stored Python objects retain their own thread-safety requirements.

## Documentation

Canonical documentation is maintained on `spork.sh`:

- [Package overview](https://spork.sh/docs/packages/spork-pds/)
- [Practical guide](https://spork.sh/docs/packages/spork-pds/guide/)
- [API reference](https://spork.sh/docs/packages/spork-pds/api/)
- [Design and complexity](https://spork.sh/docs/packages/spork-pds/design/)
- [Native free-threading](https://spork.sh/docs/packages/spork-pds/free-threading/)
- [Benchmark methodology](https://spork.sh/docs/packages/spork-pds/benchmarks/)
- [Changelog](CHANGELOG.md)

## Development

```bash
git clone https://github.com/spork-it/spork-pds.git
cd spork-pds
make venv
make test
make fuzz
```

Use `make help` for build, sanitizer, benchmark, and distribution targets.

## License

MIT. See [LICENSE](LICENSE).
