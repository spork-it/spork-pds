# Native Free-Threading Support

`spork-pds` declares `Py_MOD_GIL_NOT_USED` on CPython 3.13 and newer. A
free-threaded CPython 3.14t process therefore keeps the GIL disabled when it
imports either `spork_pds` or `spork.pds`.

## Supported contract

- Persistent values are immutable after publication and may be shared between
  threads.
- Different transient builders may execute in parallel, including builders
  created from the same persistent value.
- Each transient is confined to the Python thread that created it on a
  free-threaded build. Wrong-thread access raises `RuntimeError` before mutable
  state is read or changed.
- Iterator cursor mutation and lazy hash/buffer publication are synchronized.
- Objects stored inside a collection retain their own synchronization
  requirements.
- Isolated and per-interpreter-GIL subinterpreters are not supported.

Concurrent mutation of one transient is intentionally outside this contract.
Convert a transient to a persistent value before passing it to another thread.

## Validation tools

The normal test suite contains semantic concurrency tests. Two additional
standalone tools make release validation reproducible:

```bash
# Thousands of synchronized cache publications plus typed-history, HAMT,
# and cyclic-GC churn. Add --require-no-gil for a 3.14t release run.
python tools/stress_free_threading.py --require-no-gil

# Synchronization-sensitive microbenchmarks with optional JSON output.
python tools/benchmark_free_threading.py \
  --size 4096 --repeats 9 --workers 8 \
  --json free-threaded-results.json
```

`stress_free_threading.py` uses only the standard library, so it also runs
under a custom sanitizer-instrumented CPython without installing pytest. Its
default publication test performs 1,000 barrier rounds across up to eight
workers (8,000 synchronized worker starts), then exercises retained typed
vector histories, independent dense/collision HAMT transients, and concurrent
cycle collection.

The release test workflow additionally runs:

- the full suite and stress runner with ASan and UBSan;
- the stress runner with a free-threaded CPython built using ThreadSanitizer;
- `PYTHONMALLOC=debug` and `-X dev` tests;
- vector fuzzing under CPython 3.14t.

### Dynamic-analysis record

The native implementation at `21c4519` was validated on 2026-08-30 on an
8-core Apple M1 running macOS 26.3.1:

| Instrumentation | Interpreter/toolchain | Result |
| --- | --- | --- |
| ThreadSanitizer | CPython 3.14.7 free-threaded debug build; Apple Clang 21 | Installed-import smoke and the default stress profile passed, including 8,000 synchronized publication starts. No `spork-pds` race report was emitted. |
| AddressSanitizer + UBSan | CPython 3.14.4t; extension built with Apple Clang 21 | `147 passed, 1 skipped`; the default stress profile also passed under `PYTHONMALLOC=debug` and `-X dev`. |
| UBSan | CPython 3.14t native no-GIL build | Full tests and repeated concurrency stress passed without a diagnostic. |

ThreadSanitizer uses CPython's version-matched
`Tools/tsan/suppressions_free_threading.txt`. There are no project-level
suppressions and no suppression naming a `spork-pds` function. Those upstream
suppressions cover known reports in the interpreter/toolchain rather than
extension state. CI builds CPython itself with `--disable-gil`, `--with-pydebug`,
and `--with-thread-sanitizer`; instrumenting only the extension would not make
all interpreter synchronization visible to ThreadSanitizer.

LeakSanitizer is disabled in the ASan job because the extension is loaded into
a packaged, non-ASan CPython process with process-lifetime interpreter
allocations. Address and undefined-behavior checks remain fatal. On macOS,
`SPORK_PDS_DYLD_INSERT_LIBRARIES` is used only to propagate the ASan runtime to
subprocess tests after dyld removes its control variable from the parent
environment.

## Performance record

Run regular and free-threaded interpreters separately; build-mode differences
make direct cross-mode ratios misleading. The benchmark can compare JSON
results only when the mode and configuration match.

A regular CPython 3.14.7 build at `21c4519` was compared on the host above with
the pre-synchronization regular-GIL baseline `9612977`, using size 4,096, nine
samples, and eight workers. Median current/baseline ratios were:

| Operation | Ratio |
| --- | ---: |
| Uncached / cached persistent hash | 1.037x / 1.003x |
| Vector index / iteration | 1.014x / 1.010x |
| Map lookup | 0.975x |
| Persistent vector `conj` / `assoc` | 1.033x / 0.988x |
| Persistent map `assoc` | 0.951x |
| Vector / map transient build | 1.031x / 1.016x |
| First / repeated typed-buffer export | 1.023x / 1.001x |

The largest observed slowdown was 3.7% in uncached hashing; cache hits and
non-cache reads were effectively unchanged at this measurement resolution.
No unexplained regression was found.

The separate CPython 3.14.4t native no-GIL run measured a 2.75x elapsed-time
speedup for eight independent vector/map transient workloads compared with
executing the same eight workloads serially. The matching regular-GIL run was
1.10x. This is a workload result, not a universal scaling guarantee; nested
Python callbacks, core count, allocator behavior, and input types can change
scaling.

To compare two revisions on the same interpreter and host:

```bash
python tools/benchmark_free_threading.py \
  --size 4096 --repeats 9 --workers 8 \
  --json current.json \
  --baseline baseline.json
```

Use `--max-regression 1.10` when a controlled environment is stable enough to
make a 10% threshold meaningful. The option is intentionally not enabled in
normal CI because shared hosted runners are too noisy for reliable performance
gating.

## Wheels and source builds

Free-threaded wheels are separate `cp314-cp314t` artifacts; a regular
`cp314-cp314` wheel cannot be reused. The release workflow builds both modes on
supported Linux, macOS, Windows, x86-64, and ARM64 runners. Every wheel is
installed into cibuildwheel's isolated test environment and checked with
`tools/smoke_installed_distribution.py`. The free-threaded smoke runs once at
default startup and once with `PYTHON_GIL=0`, asserting that imports leave the
GIL disabled.

`tools/verify_distributions.py` rejects mixed or missing regular/free-threaded
ABI tags and verifies that the native extension filename carries the `t`
suffix. The aggregate release job runs these checks and `twine check` before
publishing. Source distributions are installed in a fresh environment and
smoke-tested as well.

Official free-threaded Windows installs do not automatically define
`Py_GIL_DISABLED` for extension compilation. `setup.py` derives the build mode
from `sysconfig` and explicitly defines the macro there.
