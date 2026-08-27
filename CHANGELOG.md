# Changelog

All notable changes to `spork-pds` will be documented here.

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
