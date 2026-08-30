# Persistent Data Structure Benchmarks

## Overview

The benchmark suite compares the persistent collections exposed by `spork.pds` with Python's built-in mutable collections. The public module is a thin facade over the native extension, so no Spork compiler or runtime is involved.

The suite was ported from `spork-lang/tools/benchmark_pds.py` when the extension became its own project.

This document describes methodology rather than publishing a permanent winner table. Results depend heavily on the host and workload; generate a report when you need a comparable snapshot.

## Contents

- [What it measures](#what-it-measures)
- [Methodology](#methodology)
- [Running the suite](#running-the-suite)
- [Reading results](#reading-results)
- [Generating a report](#generating-a-report)
- [Free-threading benchmarks](#free-threading-benchmarks)
- [Adding benchmarks](#adding-benchmarks)

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

Python built-ins are mutable, while `spork-pds` values preserve old versions. Raw timings must be interpreted together with those semantic differences. A persistent update is most directly comparable to copying a built-in collection and then mutating the copy when both workloads must preserve the old version.

The runner is intentionally lightweight rather than a statistical benchmarking framework: it reports one mean from one process. Use multiple fresh processes and inspect variance before drawing small-difference conclusions.

## Running the suite

Run commands from the repository root. Set up and build the project first:

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

The equivalent Make target builds the extension first:

```bash
make benchmark BENCH_ARGS="--size 100000 --iter 50 --seed 0"
```

Set `NO_COLOR=1` for plain output when saving terminal results:

```bash
NO_COLOR=1 make benchmark BENCH_ARGS="--size 100000 --iter 50 --seed 0"
```

## Reading results

Each group is sorted from fastest to slowest. Ratios are relative to the fastest case in that group, not necessarily to Python's built-in collection. `~same` means the measured ratio falls within the runner's display threshold; it is not a claim of statistical equivalence.

Before comparing two rows, check:

- whether both operations preserve the same previous versions;
- whether construction includes input conversion or starts from prepared data;
- whether a transient or persistent update matches the intended application pattern;
- whether a typed-vector result pays its first buffer-materialization cost;
- whether the collection size is representative;
- whether the difference persists across repeated processes and realistic data.

Microbenchmarks isolate operation overhead. They do not predict end-to-end application performance, memory use, cache behavior, or contention by themselves.

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

Generated reports are snapshots for one machine and environment. The report includes host information, package version, sizes, iteration count, seed, and raw output. Keep all of that context with any published result, along with compiler flags or environment details that the host report does not capture.

For comparisons across commits, use the same machine and power mode, rebuild the extension for each commit, close unrelated workloads, and run the report more than once. Avoid comparing generated reports from different Python build modes as though only the `spork-pds` code changed.

## Free-threading benchmarks

`tools/benchmark_free_threading.py` focuses on paths affected by the native free-threading design:

- cached and uncached persistent hashes;
- vector indexing and iteration;
- map lookup;
- persistent vector `conj`/`assoc` and map `assoc`;
- single-thread vector/map transient construction;
- serial and parallel independent transient construction;
- first and repeated typed-buffer export.

Run it separately with regular and free-threaded interpreters:

```bash
python tools/benchmark_free_threading.py \
  --size 4096 --repeats 9 --workers 8 \
  --json results.json
```

The equivalent Make target is:

```bash
make benchmark-free-threading \
  FT_BENCH_ARGS="--size 4096 --repeats 9 --workers 8 --json results.json"
```

A JSON report may be compared only with a report using the same build mode and configuration:

```bash
python tools/benchmark_free_threading.py \
  --size 4096 --repeats 9 --workers 8 \
  --baseline baseline.json \
  --json current.json
```

`--max-regression 1.10` can make a controlled comparison fail above a 10% median regression. Do not use a tight threshold on shared CI hosts, and do not compare regular and free-threaded results as if their interpreter builds were otherwise identical.

The native free-threading validation methodology and recorded release measurements are in [Native Free-Threading Support](FREE_THREADING.md).

## Adding benchmarks

Add focused methods to `Benchmarks` in `tools/benchmark_pds.py` and call them from the corresponding section in `main()`. A useful comparison should:

- return or consume its result so the operation is actually performed;
- prepare reusable data outside the timed function when setup is not under test;
- compare operations with clearly stated semantics;
- avoid combining unrelated work in one timing;
- remain practical at the default sizes;
- use deterministic input from the suite's seeded random generator where randomness is needed;
- label persistent, transient, copying, and mutating cases so their semantics are visible.

When setup is part of the operation under test, say so in the case name. When it is not, build inputs in `Benchmarks.__init__` or another untimed preparation step.
