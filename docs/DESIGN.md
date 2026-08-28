# Design and Complexity

This document describes the implementation model and expected asymptotic costs. For public signatures, see the [API Reference](API.md); for usage patterns, see the [Practical Guide](GUIDE.md).

## Persistence and structural sharing

A persistent collection never changes after publication. Operations such as `assoc`, `conj`, `dissoc`, and `pop` return a value with the requested change.

The implementation copies only nodes along the changed path. Unaffected branches remain shared:

```text
old root ──┬── shared branch A
           └── old branch B

new root ──┬── shared branch A
           └── new branch B
```

This is structural sharing. It makes retaining snapshots and branching histories much cheaper than copying an entire collection for every update.

The important observable properties are:

- an old collection does not reflect updates made through a new one;
- persistent operations do not mutate either operand;
- equal values may have different internal node layouts;
- object identity and the amount of sharing are implementation details.

Immutability is shallow. The collection prevents replacement of its slots, but an element that is itself mutable remains mutable through any other reference to it.

## Collection internals

### Vector

`Vector` is a 32-way bit-partitioned trie with a tail block. Index paths consume five index bits per level. The tail keeps normal append workloads from requiring a new tree path for every element.

- indexed reads follow one shallow trie path;
- `assoc` copies the path to one leaf;
- `conj` usually updates the tail and occasionally promotes a full tail into the trie;
- `pop` performs the reverse operation and can reduce tree depth;
- slicing creates a persistent result containing the selected values.

`DoubleVector` and `IntVector` use the same broad trie shape with unboxed `double` and signed 64-bit values. Trie leaves are not one contiguous allocation, so the buffer protocol lazily materializes a contiguous cache. The cache is safe to reuse because the vector cannot change.

### Map and Set

`Map` is a hash array mapped trie (HAMT). A key's hash is consumed in five-bit chunks to choose branches. Nodes can use compact bitmap layouts, dense arrays, or collision storage depending on occupancy and hash behavior.

`Set` reuses the HAMT machinery while storing keys without separate mapped values.

Hashing and equality of Python objects remain part of the operation cost. Adversarial collisions or expensive user-defined `__hash__` and `__eq__` methods can dominate the shallow trie traversal.

### SortedVector

`SortedVector` is a persistent red-black tree. Every node stores its subtree size in addition to color and child links.

Red-black balancing keeps tree height logarithmic. Subtree sizes allow indexed lookup and rank queries without flattening the tree. Insertion, deletion, and rebalancing copy only affected paths.

The configured `key` and `reverse` policy are part of the collection's ordering behavior and are retained by updates and transients. Duplicates are stored as separate values.

### Cons

`Cons` is a conventional immutable linked-list cell with `first` and `rest` references. Prepending creates one cell and shares the entire prior list. Full traversal is linear.

## Expected operation costs

`n` is the current collection size, `m` is the number of incoming or selected values, and `k` is a repetition count. `log n` means the shallow trie or tree depth; for hash tries the expected base is 32.

| Operation | Vector | Map | Set | SortedVector |
| --- | ---: | ---: | ---: | ---: |
| Length | O(1) | O(1) | O(1) | O(1) |
| Indexed lookup | O(log n) | — | — | O(log n) |
| Key lookup / membership | O(n) membership | expected O(log n) | expected O(log n) | O(log n) |
| Persistent add | amortized O(1) | expected O(log n) | expected O(log n) | O(log n) |
| Persistent update | O(log n) | expected O(log n) | — | — |
| Persistent removal | amortized O(1) from end | expected O(log n) | expected O(log n) | O(log n) |
| Concatenate / merge / set operation | O(m) | expected O(m log n) | expected O(n + m) | — |
| Repetition | O(nk) | — | — | — |
| Slice of `m` values | O(m log n) | — | — | — |
| Full iteration | O(n) | O(n) | O(n) | O(n) |
| Typed-vector first buffer request | O(n) | — | — | — |
| Typed-vector later buffer request | O(1) | — | — | — |

These bounds describe collection mechanics, not every possible user callback. Sorting also incurs comparison or key-function costs. Hash-trie bounds are expected rather than adversarial worst-case bounds.

The implementation may return the original object for a no-op such as associating an existing value or removing a missing key. Treat that as an optimization, not as a required identity guarantee.

## Transients

Persistent path copying is unnecessary overhead when constructing one final value through many intermediate edits. A transient gives a collection an edit token. Nodes owned by that token can be modified in place; shared or unowned nodes are copied before editing.

The intended lifecycle is:

```text
persistent value → transient builder → persistent result
                         │
                         └── invalid after conversion
```

Calling `persistent()` removes editability and invalidates the transient. Subsequent reads as well as writes raise `RuntimeError`. This strict boundary prevents mutable state from leaking into a published persistent value.

Transients are local, single-owner builders. They are not designed for long-lived mutation, version history, or sharing among threads. The general vector, map, and set transients implement Python mutable collection ABCs; typed and sorted transients expose smaller operation sets.

Persistent binary operators use transients internally where useful to construct one result efficiently. Neither operand is mutated.

## Equality, hashing, and identity

Persistent values use value-oriented equality where documented by the [API Reference](API.md). `Map` and `Set` equality are independent of iteration order; `Vector`, `SortedVector`, and `Cons` equality are order-sensitive.

Hashing requires all participating nested values to be hashable. The hash contract follows Python's requirement that equal hashable values have equal hashes. Mutating a Python object nested inside a persistent collection can therefore make value and hash behavior unsafe; use immutable nested values when the collection itself will be a key or set member.

`.copy()` on `Vector`, `Map`, and `Set` returns the same object because there is no mutable container state to duplicate. Code should still rely on value semantics rather than identity.

## Python protocol integration

The extension registers persistent collections with `collections.abc`:

- vectors, sorted vectors, and cons cells as `Sequence`;
- maps as `Mapping`;
- sets as `Set`.

`TransientVector`, `TransientMap`, and `TransientSet` register with the corresponding mutable ABCs. Types also participate directly in iteration, hashing, comparison, indexing, mapping, set-operation, generic-alias, pickle, and buffer protocols as applicable.

No in-place numeric slots are installed. Augmented assignment therefore uses Python's immutable fallback:

```python
updated = original
updated |= changes   # binary operation, then rebinding
```

`original` remains unchanged.

## Memory and lifetime considerations

Keeping many versions is cheaper than retaining full copies, but it is not free. Every update allocates at least a new root and usually a small path of nodes. A shared branch remains alive as long as any version references it.

The first buffer view of a typed vector adds a contiguous cache proportional to the vector size. That cache remains associated with the immutable vector while the vector is alive. This trades memory for constant-time subsequent buffer requests.

Pickle stores values and ordering configuration, not a promise about trie shape or sharing relationships after deserialization.

## Free-threaded CPython

On free-threaded CPython 3.14+, the extension declares that it does not require the GIL. Persistent collection nodes are immutable after publication, which makes them suitable for sharing between readers.

This does not make arbitrary nested Python objects thread-safe, and it does not make transients safe for concurrent access. Applications remain responsible for the synchronization requirements of contained values and for keeping each transient under one owner's control.
