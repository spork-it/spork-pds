# API Reference

The distribution name is `spork-pds`; the import name is `spork_pds`.

```python
import spork_pds
```

## Module exports

### Factory functions

| Function | Result | Notes |
| --- | --- | --- |
| `vec(*values)` | `Vector` | With one non-string iterable, consumes that iterable. With multiple arguments, treats them as elements. |
| `vec_f64(*values)` | `DoubleVector` | Converts positional numeric values to C `double`. |
| `vec_i64(*values)` | `IntVector` | Converts positional integer values to signed 64-bit integers. |
| `hash_map(*key_values)` | `Map` | Accepts alternating keys and values. The argument count must be even. |
| `hash_set(iterable=None)` | `Set` | Builds a set from at most one iterable. |
| `sorted_vec(iterable=None, *, key=None, reverse=False)` | `SortedVector` | Builds an ordered collection; duplicates are retained. |
| `cons(first, rest=None)` | `Cons` | Creates one immutable linked-list cell. |

### Types

Persistent types:

- `Vector`
- `DoubleVector`
- `IntVector`
- `Map`
- `Set`
- `SortedVector`
- `Cons`

Transient types:

- `TransientVector`
- `TransientDoubleVector`
- `TransientIntVector`
- `TransientMap`
- `TransientSet`
- `TransientSortedVector`

Persistent `Vector`, `Map`, and `Set` values can be constructed directly from Python iterables or mappings. Prefer `.transient()` over constructing transient classes directly. The factory functions remain available for compatibility and specialized construction.

### Empty values

- `EMPTY_VECTOR`
- `EMPTY_DOUBLE_VECTOR`
- `EMPTY_LONG_VECTOR`
- `EMPTY_MAP`
- `EMPTY_SET`
- `EMPTY_SORTED_VECTOR`

The `vec`, `vec_f64`, `vec_i64`, `hash_map`, and `hash_set` empty factories return their shared empty values. `sorted_vec()` creates an equivalent empty sorted vector. Empty values are immutable and safe to reuse.

## Persistent operators

The primary collection operators follow Python's built-in collection vocabulary:

| Expression | Result |
| --- | --- |
| `vector + iterable` | Concatenated `Vector` |
| `vector * count`, `count * vector` | Repeated `Vector` |
| `map_value | mapping` | Merged `Map`; right-hand values win |
| `mapping | map_value` | Merged `Map`; right-hand values win |
| `set_value | other` | Union as a `Set` |
| `set_value & other` | Intersection as a `Set` |
| `set_value - other` | Difference as a `Set` |
| `set_value ^ other` | Symmetric difference as a `Set` |

Built-in `set` and `frozenset` values are also supported on the left of set operators; the result is still a persistent `Set`. Iterable-left vector concatenation is intentionally unsupported, so `[1] + vector` raises `TypeError`.

No in-place number slots are defined. Augmented assignment therefore computes a persistent result and rebinds the target name:

```python
original = Map({"count": 1})
updated = original
updated |= {"count": 2}

assert original["count"] == 1
assert updated["count"] == 2
```

## `Vector`

A general-purpose persistent sequence. `Vector()` accepts at most one iterable; use `vec(*values)` when variadic element construction is convenient.

```python
from spork_pds import Vector

original = Vector([10, 20, 30])
appended = original + [40]
repeated = original * 2
updated = original.assoc(1, 99)
shorter = appended.pop()
slice_value = appended[1:3]
```

Methods:

- `nth(index, default=...)`: return an element. If a default is provided, an out-of-range index returns it; otherwise raises `IndexError`.
- `conj(value)`: return a vector with `value` appended.
- `assoc(index, value)`: return a vector with one index replaced. Associating at `len(vector)` appends.
- `pop()`: return a vector without its final value.
- `transient()`: create a `TransientVector` for batch updates.
- `to_seq()`: convert to a `Cons` chain, or `None` when empty.
- `copy()`: return the same object because vectors are immutable.
- `index(value, start=0, stop=len(vector))`: find the first matching index.
- `count(value)`: count matching values.
- `sort(*, key=None, reverse=False)`: return a sorted `Vector`.

Python support includes `len`, iteration, `reversed`, membership, integer indexing, negative indexing, slicing, `+` concatenation, `*` repetition, equality, hashing, `collections.abc.Sequence`, generic aliases such as `Vector[int]`, and pickle. Concatenation accepts any iterable on the right. Repetition accepts an integer-like count; non-positive counts produce the empty vector.

### `TransientVector`

Methods:

- `conj_mut(value)` / `append(value)`
- `assoc_mut(index, value)`
- `pop_mut()`
- `extend(iterable)`
- `sort(*, key=None, reverse=False)`
- `persistent()`

It also supports length, iteration, membership, indexing, and indexed assignment while editable.

```python
transient = vec(3, 1).transient()
transient.append(2)
transient.sort()
result = transient.persistent()
assert list(result) == [1, 2, 3]
```

## `DoubleVector` and `IntVector`

Specialized persistent sequences store unboxed C values:

- `DoubleVector`: float64 (`double`)
- `IntVector`: signed int64 (`int64_t`)

Both support `len`, iteration, indexing, negative indexing, slicing, hashing, `nth`, `conj`, `transient`, generic aliases, pickle, and `collections.abc.Sequence`.

Their buffers are one-dimensional and read-only:

```python
from spork_pds import vec_f64, vec_i64

floats = vec_f64(1, 2.5, 3)
integers = vec_i64(1, 2, 3)

float_buffer = memoryview(floats)
int_buffer = memoryview(integers)

assert float_buffer.format == "d"
assert int_buffer.format == "q"
assert float_buffer.readonly and int_buffer.readonly
```

A first buffer request flattens the trie into cached contiguous storage. Further views of the same immutable value reuse that storage.

`TransientDoubleVector` and `TransientIntVector` expose `conj_mut(value)` and `persistent()`.

## `Map`

A persistent hash map.

```python
from spork_pds import Map

original = Map({"a": 1, "b": 2})
updated = original | {"b": 20, "c": 3}
removed = updated.dissoc("a")

assert original["b"] == 2
assert updated["b"] == 20
assert dict(removed.items()) == {"b": 20, "c": 3}
```

`Map()` accepts at most one mapping or iterable of key/value pairs, followed by optional keyword entries. `Map | Mapping` and `Mapping | Map` use dict-union precedence: values from the right operand win, and the result is always a persistent `Map`.

Methods:

- `get(key, default=None)`
- `assoc(key, value)`
- `dissoc(key)`
- `keys()`, `values()`, `items()`
- `transient()`
- `to_seq()`: return a `Cons` chain of key/value tuples.
- `copy()`: return the same immutable object.

Maps support `len`, key iteration, membership, subscription, `|` merge, equality, hashing, `collections.abc.Mapping`, generic aliases such as `Map[str, int]`, and pickle.

### `TransientMap`

Methods:

- `assoc_mut(key, value)`
- `dissoc_mut(key)`
- `get(key, default=None)`
- `keys()`, `values()`, `items()`
- `persistent()`

It also supports the mutable mapping protocols for lookup, assignment, deletion, iteration, membership, and length.

```python
transient = hash_map("a", 1).transient()
transient["b"] = 2
del transient["a"]
result = transient.persistent()
```

## `Set`

A persistent hash set.

```python
from spork_pds import Set

left = Set([1, 2, 3])
right = Set([3, 4])

assert set(left | right) == {1, 2, 3, 4}
assert set(left & right) == {3}
assert set(left - right) == {1, 2}
assert set(left ^ right) == {1, 2, 4}
```

Methods:

- `conj(value)`: add a value.
- `disj(value)`: remove a value if present.
- `transient()`
- `to_seq()`
- `copy()`: return the same immutable object.
- `isdisjoint(other)`

`Set()` accepts at most one iterable. Sets support `len`, iteration, membership, equality and ordering comparisons, hashing, standard binary set operators, reflected operations with built-in sets and frozensets, `collections.abc.Set`, generic aliases such as `Set[int]`, and pickle. Binary operators always return a persistent `Set`.

### `TransientSet`

Methods:

- `conj_mut(value)` / `add(value)`
- `disj_mut(value)` / `discard(value)`
- `remove(value)`
- `clear()`
- `persistent()`

The transient supports length, membership, and iteration while editable.

## `SortedVector`

A persistent ordered collection that retains duplicates and supports O(log n) indexing by rank.

```python
from spork_pds import sorted_vec

values = sorted_vec([3, 1, 2, 2])
assert list(values) == [1, 2, 2, 3]
assert values.first() == 1
assert values.last() == 3
assert values.rank(2) == 1
```

Methods:

- `nth(index, default=...)`
- `conj(value)`: insert while preserving order.
- `disj(value)`: remove one matching occurrence.
- `first()` / `last()`: return the first or last ordered value, or `None` when empty.
- `index_of(value)`: return an index, or `-1` when absent.
- `rank(value)`: return the insertion rank according to the configured ordering.
- `transient()`

`key` and `reverse` are preserved across persistent updates. `SortedVector` supports length, iteration, indexing, negative indexing, membership, equality, hashing, `collections.abc.Sequence`, generic aliases, and pickle. A `key` function must itself be picklable for the collection to be pickled.

`TransientSortedVector` exposes `conj_mut(value)`, `disj_mut(value)`, and `persistent()`.

## `Cons`

An immutable linked-list cell with read-only `first` and `rest` properties.

```python
from spork_pds import cons

values = cons(1, cons(2, cons(3)))
assert list(values) == [1, 2, 3]
assert values.first == 1
assert values.rest.first == 2
```

`conj(value)` prepends a new cell. Cons chains support length, iteration, equality, hashing, `collections.abc.Sequence`, generic aliases, and pickle.

## Transient lifecycle

All transient variants are single-use builders:

1. Call `.transient()` on a persistent value.
2. Apply mutable operations.
3. Call `.persistent()` once.
4. Discard the transient.

A transient is invalid after step 3. It is not a general-purpose mutable collection and should not escape the batch-update scope.
