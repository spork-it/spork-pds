# Practical Guide

This guide focuses on choosing and using `spork-pds` collections. See the [API Reference](API.md) for the complete method list and [Design and Complexity](DESIGN.md) for implementation details.

## Choose a collection

| Type | Use it for | Persistent update |
| --- | --- | --- |
| `Vector` | Ordered, indexable general-purpose values | `.conj()`, `.assoc()`, `.pop()` |
| `Map` | Hashable keys mapped to values | `.assoc()`, `.dissoc()`, `|` |
| `Set` | Unique hashable values | `.conj()`, `.disj()`, set operators |
| `SortedVector` | Values kept in order, including duplicates | `.conj()`, `.disj()` |
| `DoubleVector` | Unboxed float64 values and read-only buffers | `.conj()` |
| `IntVector` | Unboxed signed int64 values and read-only buffers | `.conj()` |
| `Cons` | Immutable linked lists and efficient prepending | `.conj()` |

Choose these collections when old versions must remain available, values need hashable collection semantics, readers share snapshots, or updates branch into multiple histories. Python's built-in mutable collections are usually simpler when there is one owner and only the latest state matters.

## Construct values

The classes accept Python-style inputs:

```python
from spork.pds import Map, Set, Vector, sorted_vec

vector = Vector(range(4))
config = Map({"host": "localhost"}, port=8000)
tags = Set(["stable", "documented", "stable"])
ordered = sorted_vec([3, 1, 2, 2])
```

Factories are convenient for variadic construction:

```python
from spork.pds import hash_map, hash_set, vec, vec_f64, vec_i64

vector = vec(10, 20, 30)
config = hash_map("host", "localhost", "port", 8000)
tags = hash_set(["stable", "documented"])
floats = vec_f64(1, 2.5, 3)
integers = vec_i64(1, 2, 3)
```

`vec` has one intentional convenience rule: one non-string iterable is consumed, while multiple arguments are treated as individual elements.

```python
assert list(vec(range(3))) == [0, 1, 2]
assert list(vec(0, 1, 2)) == [0, 1, 2]
assert list(vec("abc")) == ["abc"]
assert list(Vector("abc")) == ["a", "b", "c"]
```

Use `Vector(iterable)` when normal Python constructor behavior is clearer.

## Keep old versions

Persistent operations return a value rather than modifying the receiver:

```python
from spork.pds import Map, Vector

before = Vector([10, 20, 30])
after = before.assoc(1, 99).conj(40)

assert list(before) == [10, 20, 30]
assert list(after) == [10, 99, 30, 40]

base = Map({"host": "localhost", "port": 8000})
production = base | {"host": "example.com", "tls": True}

assert base["host"] == "localhost"
assert production["host"] == "example.com"
```

Map union follows `dict` precedence: the right operand wins. Reflected union also returns `Map`:

```python
result = {"left": 1, "shared": "left"} | Map({"shared": "right"})
assert isinstance(result, Map)
assert result["shared"] == "right"
```

Augmented assignment computes a new value and rebinds the target:

```python
current = base
current |= {"port": 443}

assert current is not base
assert base["port"] == 8000
```

Some no-op operations can return the original object as an optimization. Write code in terms of value equality and immutability rather than object identity.

## Use transients for batches

Repeated persistent updates preserve every intermediate version. When only the final result matters, use a transient builder:

```python
from spork.pds import EMPTY_VECTOR

builder = EMPTY_VECTOR.transient()
for number in range(100_000):
    builder.append(number)
values = builder.persistent()

assert values[-1] == 99_999
```

The lifecycle is strict:

1. create a transient from a persistent collection;
2. make local edits;
3. call `.persistent()` exactly once;
4. discard the transient.

After conversion, reads and writes through the transient raise `RuntimeError`.

General transients support familiar mutable Python protocols:

```python
from spork.pds import Map, Set, Vector

items = Vector([3, 1]).transient()
items.append(2)
items.extend([5, 4])
items.sort()
sorted_items = items.persistent()

config = Map({"host": "localhost"}).transient()
config["port"] = 8000
del config["host"]
final_config = config.persistent()

tags = Set(["draft"]).transient()
tags.add("ready")
tags.discard("draft")
final_tags = tags.persistent()
```

`TransientVector`, `TransientMap`, and `TransientSet` integrate with `MutableSequence`, `MutableMapping`, and `MutableSet`. Typed-vector and sorted-vector transients intentionally expose only their focused `conj_mut`, `disj_mut`, and `persistent` operations as applicable.

Do not use a transient as a long-lived mutable collection or share one among threads.

## Work with sorted values

`SortedVector` retains duplicates and stores its ordering policy with the value:

```python
from spork.pds import sorted_vec

scores = sorted_vec([30, 10, 20, 20], reverse=True)
updated = scores.conj(25)

assert list(scores) == [30, 20, 20, 10]
assert list(updated) == [30, 25, 20, 20, 10]
```

Use `key` to sort records without changing the stored values:

```python
rows = [
    {"name": "one", "score": 10},
    {"name": "two", "score": 30},
    {"name": "three", "score": 20},
]
ranking = sorted_vec(rows, key=lambda row: row["score"], reverse=True)
assert [row["name"] for row in ranking] == ["two", "three", "one"]
```

The key function and `reverse` flag are retained by `.conj()`, `.disj()`, and `.transient()`. A keyed sorted vector can be pickled only when its key function is picklable; a named top-level function or a function from `operator` is safer than a lambda for persisted data.

## Share numeric data through buffers

`DoubleVector` and `IntVector` store unboxed values and export one-dimensional, read-only buffers:

```python
from spork.pds import vec_f64, vec_i64

floats = vec_f64(1, 2.5, 3)
integers = vec_i64(1, 2, 3)

assert memoryview(floats).format == "d"
assert memoryview(integers).format == "q"
```

NumPy can use those buffers without requesting a writable copy:

```python
import numpy as np

array = np.asarray(floats)
assert array.tolist() == [1.0, 2.5, 3.0]
assert not array.flags.writeable
```

The first buffer request creates and caches contiguous storage for that immutable vector. Later views reuse the cache. Use `np.array(floats, copy=True)` when a writable array is required.

`IntVector` accepts values representable by a signed 64-bit integer. `DoubleVector` converts numeric input to C `double`.

## Convert at boundaries

All persistent types are iterable and register with appropriate `collections.abc` interfaces:

```python
from collections.abc import Mapping, Sequence, Set as AbstractSet

assert isinstance(Vector(), Sequence)
assert isinstance(Map(), Mapping)
assert isinstance(Set(), AbstractSet)
```

Convert explicitly when an API requires an exact built-in type:

```python
python_list = list(vector)
python_dict = dict(config.items())
python_set = set(tags)
```

Convert results back with constructors or factories. Conversion creates a new collection representation; it does not make nested values immutable.

## Hashing and shallow immutability

Persistent collections do not let callers replace their elements in place, but immutability is shallow. A mutable object stored inside a vector or map can still be changed through another reference:

```python
payload = []
value = Vector([payload])
payload.append("changed")
assert value[0] == ["changed"]
```

Likewise, a persistent collection is hashable only when all content needed for its hash is hashable. Prefer immutable nested values when using collections as dictionary keys or set members.

## Pickle and processes

Persistent values support pickle:

```python
import pickle

restored = pickle.loads(pickle.dumps(vector))
assert restored == vector
```

Pickle recreates the same public type and contents; object identity and internal sharing relationships are not an API guarantee. Follow normal pickle security rules and never unpickle untrusted data.

## Common pitfalls

- **Expecting mutation:** assign the result of every persistent update.
- **Keeping a transient after `.persistent()`:** conversion invalidates it immediately.
- **Using unhashable map keys or set members:** keys and members follow Python's hashability rules.
- **Assuming deep immutability:** nested Python objects retain their own mutation behavior.
- **Using a lambda as a persisted sort key:** lambdas are generally not picklable.
- **Treating benchmark ratios as universal:** compare workloads and semantics, not only one timing. See [Benchmarks](BENCHMARKS.md).
