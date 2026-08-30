# Changelog

All notable changes to `spork-pds` will be documented here.

## Unreleased

- Integrate object-bearing collections, internal nodes, transients, and iterators with Python's cyclic garbage collector.
- Fix typed-vector node ownership across structurally shared versions.
- Harden singleton, temporary-builder, iterator, allocation, and hash-overflow behavior.
- Add native free-threading support for CPython 3.14t without enabling the compatibility GIL.
- Synchronize lazy hash/buffer publication and iterator cursor advancement.
- Confine each transient to its creating thread on free-threaded builds while allowing independent builders to run in parallel.
- Hold strong references while unpacking externally supplied mutable map pairs.
- Add sanitizer-backed no-GIL stress and synchronization-sensitive performance validation.
- Build, install, and smoke-test ABI-separated CPython 3.14t wheels before release publication.

## 0.1.3

- Split the native extension into focused C translation units under `src/`.

## 0.1.2

- Add `spork.pds` as the canonical public import namespace.
- Retain the native `spork_pds` module for import, pickle, and type-name compatibility.

## 0.1.1

- Expand API, boundary, collision, transient lifecycle, and model-based test coverage.
- Fix `SortedVector` queries and removals for duplicate and equal sort keys.
- Prevent repeated transient sorted-vector edits from corrupting shared tree structure.
- Reject unsupported keyword arguments in typed-vector constructors.
- Enforce transient invalidation for length checks.

## 0.1.0

- Extract the persistent data structure extension from `spork-lang`.
- Add Pythonic `Vector(iterable)`, `Map(mapping, **kwargs)`, and `Set(iterable)` constructors.
- Add `Vector * count` and `count * Vector` repetition.
- Support reflected map merge and built-in-set operations with persistent results.
- Document operators as the primary persistent collection interface.
- Fix unsafe reflected vector addition that could dereference a non-vector operand.
- Expose the extension as the standalone `spork_pds` module.
- Add Python API tests, vector fuzzing, and cross-platform CI.
- Port the direct C-extension benchmark and report-generation tools.
- Add standalone API, design, benchmark, build, and usage documentation.
