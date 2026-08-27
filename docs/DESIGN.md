# Design and Complexity

## Persistence and structural sharing

A persistent collection never changes after it has been published. Operations such as `assoc`, `conj`, `dissoc`, and `pop` return a new collection. The new and old values share all internal nodes not affected by the update.

This gives two useful properties:

- retaining an old version is inexpensive;
- code holding an old value cannot observe updates made through a newer value.

The tradeoff is additional pointer traversal and allocation compared with Python's mutable built-ins. Persistent collections are most valuable when snapshots, branching histories, value semantics, or concurrent readers matter.

## Collection internals

### Vector

`Vector` is a 32-way bit-partitioned trie with a tail block. Index paths consume five hash/index bits per level. Appends are amortized O(1), while random lookup and path-copying updates are O(log32 n).

`DoubleVector` and `IntVector` use the same broad trie shape with unboxed primitive values. Their read-only buffer implementation lazily creates a contiguous cache because trie leaves are not themselves one contiguous allocation.

### Map and Set

`Map` is a hash array mapped trie (HAMT). Nodes can be bitmap-indexed, dense arrays, or hash-collision nodes. `Set` reuses the same HAMT machinery while storing keys only.

Expected lookup, insertion, and deletion are effectively O(log32 n), subject to Python hashing and equality costs. Path copying preserves older versions.

### SortedVector

`SortedVector` is a red-black tree with subtree-size annotations. The size metadata supports indexed lookup and rank operations without flattening the tree. Insertion and deletion preserve ordering and rebalance through path copying.

### Cons

`Cons` is a conventional immutable linked-list cell. Prepending is O(1); indexing by traversal is not exposed directly.

## Expected operation costs

`log n` below means the shallow trie/tree depth, not a full collection copy.

| Operation | Vector | Map | Set | SortedVector |
| --- | ---: | ---: | ---: | ---: |
| Length | O(1) | O(1) | O(1) | O(1) |
| Indexed lookup | O(log n) | — | — | O(log n) |
| Key lookup / membership | O(n) membership | expected O(log n) | expected O(log n) | O(log n) |
| Persistent add | amortized O(1) | expected O(log n) | expected O(log n) | O(log n) |
| Persistent update | O(log n) | expected O(log n) | — | — |
| Persistent removal | amortized O(1) from end | expected O(log n) | expected O(log n) | O(log n) |
| Full iteration | O(n) | O(n) | O(n) | O(n) |
| Typed-vector first buffer request | O(n) | — | — | — |
| Typed-vector later buffer request | O(1) | — | — | — |

## Transients

Persistent path copying is unnecessary overhead when constructing one final value through many intermediate changes. A transient grants a unique edit token to a collection's mutable path. Nodes owned by that token can be edited in place; unowned nodes are copied before editing.

Calling `persistent()` removes the token and invalidates the transient. This boundary is what makes the resulting collection safe to treat as immutable.

Transients are intended for local, single-owner batches. Do not share one transient across threads or retain it after conversion.

## Python integration

The extension registers persistent collections with `collections.abc`:

- vectors, sorted vectors, and cons cells as `Sequence`;
- maps as `Mapping`;
- sets as `Set`.

General transients register with their corresponding mutable ABCs. The types also participate directly in Python's iteration, hashing, comparison, indexing, mapping, set-operation, pickle, generic-alias, and buffer protocols as applicable.

On free-threaded CPython 3.13+, the extension declares that it does not require the GIL. Shared persistent values are immutable; mutable transient values still require application-level single-owner discipline.

## Source layout

The implementation intentionally remains in one file, [`pds.c`](../pds.c). Packaging, tests, documentation, and benchmark utilities live alongside it, but the C implementation is not split across translation units.
