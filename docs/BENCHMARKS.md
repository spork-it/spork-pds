# Persistent Data Structure Benchmarks

## Overview

The benchmark suite compares the standalone `spork_pds` C extension with Python's built-in mutable collections. It imports the extension directly, so no Spork compiler or runtime is involved.

The suite was ported from `spork-lang/tools/benchmark_pds.py` when the extension became its own project.

## What it measures

### Vectors

- construction through Python lists, persistent updates, transients, and factories;
- float64 and int64 specialized-vector construction;
- random access and sequential iteration;
- persistent and transient pop operations.

### Maps

- construction through dictionaries, persistent updates, transients, and `hash_map`;
- successful and missing-key lookup;
- persistent and transient removal;
- key, value, and item iteration.

### Sets

- construction;
- membership;
- persistent and transient removal;
- iteration.

### Structural sharing

These cases compare a full Python collection copy followed by mutation with a path-copying persistent update. They cover both one update and a chain of updates.

### Utilities and NumPy

- `len` and eager `to_seq()` conversion;
- optional NumPy array creation and reductions over `DoubleVector`'s buffer.

The first `DoubleVector` buffer request creates its contiguous cache. The NumPy timing measures subsequent views of that cached immutable storage.

## Methodology

For each case the runner:

1. performs one untimed warm-up;
2. runs garbage collection;
3. disables cyclic garbage collection during timing;
4. executes the requested number of iterations;
5. reports the mean duration;
6. orders each comparison from fastest to slowest.

The default random seed is fixed so access and update workloads are reproducible. Results still vary with CPU, compiler, Python build, allocator, thermal state, background load, and collection size.

Python built-ins are mutable, while `spork-pds` values preserve old versions. Raw timings should be interpreted together with those semantic differences.

## Running the suite

Set up and build the project first:

```bash
make venv
make build
```

Run the default suite:

```bash
.venv/bin/python tools/benchmark_pds.py
```

Choose collection size, timing iterations, and random seed:

```bash
.venv/bin/python tools/benchmark_pds.py \
  --size 100000 \
  --iter 50 \
  --seed 0
```

NumPy comparisons run when NumPy is installed and are skipped otherwise. The development extra installs NumPy.

The equivalent Make target is:

```bash
make benchmark BENCH_ARGS="--size 100000 --iter 50 --seed 0"
```

## Generating a report

`tools/generate_benchmark_report.py` runs the suite at multiple sizes and adds host information:

```bash
.venv/bin/python tools/generate_benchmark_report.py \
  --iter 50 \
  --seed 0 \
  --output benchmark-results.md \
  25000 50000 100000
```

Or:

```bash
make benchmark-report REPORT_ARGS="--iter 50 --seed 0 --output benchmark-results.md 25000 50000 100000"
```

Generated reports are snapshots for one machine and environment. Keep the command, host information, package version, sizes, iteration count, and seed with any published result.

## Adding benchmarks

Add focused methods to `Benchmarks` in `tools/benchmark_pds.py` and call them from the corresponding section in `main()`. A useful comparison should:

- return or consume its result so the operation is actually performed;
- prepare reusable data outside the timed function when setup is not under test;
- compare operations with clearly stated semantics;
- avoid combining unrelated work in one timing;
- remain practical at the default sizes.
