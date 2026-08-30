"""Benchmark synchronization-sensitive spork-pds operations.

Run this script separately with regular and free-threaded CPython builds. JSON
output can be compared against a same-build baseline with ``--baseline``.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import spork.pds as pds


SOURCES = (
    "cached and uncached persistent hashes",
    "lock-free persistent reads and updates",
    "owner-confined transient construction",
    "parallel independent transient construction",
    "first and repeated typed-buffer export",
)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def regression_limit(value):
    parsed = float(value)
    if parsed < 1.0:
        raise argparse.ArgumentTypeError("must be at least 1.0")
    return parsed


def make_cons(size):
    value = None
    for item in reversed(range(size)):
        value = pds.cons(item, value)
    return value


def make_hash_values(size, offset=0):
    data = range(offset, offset + size)
    return (
        make_cons(size),
        pds.Vector(data),
        pds.Map((item, item * 2) for item in data),
        pds.Set(data),
        pds.DoubleVector(data),
        pds.IntVector(data),
    )


def measure_case(name, unit, repeats, operations, prepare, run, cleanup=None):
    if cleanup is None:
        cleanup = lambda _context: None

    warmup_context = prepare()
    try:
        run(warmup_context)
    finally:
        cleanup(warmup_context)

    samples = []
    for _ in range(repeats):
        context = prepare()
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter_ns()
            run(context)
            elapsed = time.perf_counter_ns() - start
        finally:
            gc.enable()
            cleanup(context)
        samples.append(elapsed / operations)

    return {
        "name": name,
        "unit": unit,
        "operations_per_sample": operations,
        "median_ns_per_operation": statistics.median(samples),
        "min_ns_per_operation": min(samples),
        "samples_ns_per_operation": samples,
    }


def build_vector_transient(size, seed=0):
    builder = pds.EMPTY_VECTOR.transient()
    for value in range(size):
        builder.conj_mut(seed + value)
    result = builder.persistent()
    assert len(result) == size
    assert result[-1] == seed + size - 1
    return result


def build_map_transient(size, seed=0):
    builder = pds.EMPTY_MAP.transient()
    for value in range(size):
        builder.assoc_mut(seed + value, value)
    result = builder.persistent()
    assert len(result) == size
    assert result[seed + size - 1] == size - 1
    return result


def build_independent_workload(size, seed):
    return build_vector_transient(size, seed), build_map_transient(size, seed)


def run_benchmarks(size, repeats, scale, workers):
    results = []
    hash_size = min(size, 256)
    hash_batches = 24 * scale
    cached_hash_loops = 5000 * scale
    read_loops = 20 * scale
    update_count = min(size, 512) * scale
    buffer_batches = 32 * scale
    buffer_loops = 5000 * scale

    results.append(
        measure_case(
            "persistent_hash_uncached",
            "hash calls",
            repeats,
            hash_batches * 6,
            lambda: [
                make_hash_values(hash_size, batch * hash_size)
                for batch in range(hash_batches)
            ],
            lambda batches: sum(
                hash(value) for values in batches for value in values
            ),
        )
    )

    def prepare_cached_hashes():
        values = make_hash_values(hash_size)
        tuple(hash(value) for value in values)
        return values

    results.append(
        measure_case(
            "persistent_hash_cached",
            "hash calls",
            repeats,
            cached_hash_loops * 6,
            prepare_cached_hashes,
            lambda values: sum(
                hash(value) for _ in range(cached_hash_loops) for value in values
            ),
        )
    )

    vector = pds.Vector(range(size))
    indices = tuple(range(size))
    results.append(
        measure_case(
            "vector_index",
            "indexed reads",
            repeats,
            size * read_loops,
            lambda: vector,
            lambda value: sum(
                value[index] for _ in range(read_loops) for index in indices
            ),
        )
    )
    results.append(
        measure_case(
            "vector_iteration",
            "iterated values",
            repeats,
            size * read_loops,
            lambda: vector,
            lambda value: sum(item for _ in range(read_loops) for item in value),
        )
    )

    keys = tuple(range(size))
    mapping = pds.Map((key, key * 2) for key in keys)
    results.append(
        measure_case(
            "map_lookup",
            "lookups",
            repeats,
            size * read_loops,
            lambda: mapping,
            lambda value: sum(
                value[key] for _ in range(read_loops) for key in keys
            ),
        )
    )

    def persistent_vector_conj(_context):
        value = vector
        for item in range(update_count):
            value = value.conj(size + item)
        assert len(value) == size + update_count

    results.append(
        measure_case(
            "vector_persistent_conj",
            "persistent updates",
            repeats,
            update_count,
            lambda: None,
            persistent_vector_conj,
        )
    )

    assoc_indices = tuple(index % size for index in range(update_count))

    def persistent_vector_assoc(_context):
        value = vector
        for item, index in enumerate(assoc_indices):
            value = value.assoc(index, -(item + 1))
        assert value[assoc_indices[-1]] == -update_count

    results.append(
        measure_case(
            "vector_persistent_assoc",
            "persistent updates",
            repeats,
            update_count,
            lambda: None,
            persistent_vector_assoc,
        )
    )

    def persistent_map_assoc(_context):
        value = mapping
        for item in range(update_count):
            value = value.assoc(item % size, -(item + 1))
        assert value[(update_count - 1) % size] == -update_count

    results.append(
        measure_case(
            "map_persistent_assoc",
            "persistent updates",
            repeats,
            update_count,
            lambda: None,
            persistent_map_assoc,
        )
    )

    results.append(
        measure_case(
            "vector_transient_build",
            "inserted values",
            repeats,
            size,
            lambda: None,
            lambda _context: build_vector_transient(size),
        )
    )
    results.append(
        measure_case(
            "map_transient_build",
            "inserted pairs",
            repeats,
            size,
            lambda: None,
            lambda _context: build_map_transient(size),
        )
    )

    def run_serial_transients(_context):
        return [
            build_independent_workload(size, worker_id * size * 2)
            for worker_id in range(workers)
        ]

    transient_operations = workers * size * 2
    serial_result = measure_case(
        "independent_transients_serial",
        "inserted values/pairs",
        repeats,
        transient_operations,
        lambda: None,
        run_serial_transients,
    )
    results.append(serial_result)

    def prepare_parallel_transients():
        executor = ThreadPoolExecutor(max_workers=workers)
        ready = Barrier(workers + 1)
        start = Event()

        def worker(worker_id):
            ready.wait(timeout=120)
            start.wait(timeout=120)
            return build_independent_workload(size, worker_id * size * 2)

        futures = [executor.submit(worker, worker_id) for worker_id in range(workers)]
        ready.wait(timeout=120)
        return executor, start, futures

    def run_parallel_transients(context):
        _executor, start, futures = context
        start.set()
        results = [future.result(timeout=120) for future in futures]
        assert len(results) == workers

    parallel_result = measure_case(
        "independent_transients_parallel",
        "inserted values/pairs",
        repeats,
        transient_operations,
        prepare_parallel_transients,
        run_parallel_transients,
        lambda context: context[0].shutdown(wait=True),
    )
    results.append(parallel_result)

    buffer_size = min(size, 4096)

    def prepare_first_buffers():
        return [
            (pds.DoubleVector(range(buffer_size)), pds.IntVector(range(buffer_size)))
            for _ in range(buffer_batches)
        ]

    def export_first_buffers(batches):
        total = 0.0
        for double_vector, int_vector in batches:
            double_view = memoryview(double_vector)
            int_view = memoryview(int_vector)
            total += double_view[-1] + int_view[-1]
        assert total == buffer_batches * (buffer_size - 1) * 2

    results.append(
        measure_case(
            "typed_buffer_first_export",
            "buffer exports",
            repeats,
            buffer_batches * 2,
            prepare_first_buffers,
            export_first_buffers,
        )
    )

    def prepare_cached_buffers():
        values = pds.DoubleVector(range(buffer_size)), pds.IntVector(range(buffer_size))
        memoryview(values[0])
        memoryview(values[1])
        return values

    def export_cached_buffers(values):
        total = 0.0
        for _ in range(buffer_loops):
            total += memoryview(values[0])[0]
            total += memoryview(values[1])[0]
        assert total == 0

    results.append(
        measure_case(
            "typed_buffer_cached_export",
            "buffer exports",
            repeats,
            buffer_loops * 2,
            prepare_cached_buffers,
            export_cached_buffers,
        )
    )

    serial_ns = serial_result["median_ns_per_operation"]
    parallel_ns = parallel_result["median_ns_per_operation"]
    derived = {
        "independent_transient_parallel_speedup": serial_ns / parallel_ns,
    }
    return results, derived


def metadata(args):
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "configuration": {
            "size": args.size,
            "repeats": args.repeats,
            "scale": args.scale,
            "workers": args.workers,
        },
    }


def print_results(report):
    build = "free-threaded" if report["metadata"]["free_threaded"] else "regular"
    gil = "enabled" if report["metadata"]["gil_enabled"] else "disabled"
    print(f"spork-pds synchronization benchmark ({build}, GIL {gil})")
    print(f"Python: {report['metadata']['python'].splitlines()[0]}")
    print(
        f"size={report['metadata']['configuration']['size']} "
        f"repeats={report['metadata']['configuration']['repeats']} "
        f"workers={report['metadata']['configuration']['workers']}"
    )
    print()
    print(f"{'case':38} {'median ns/op':>16} {'best ns/op':>16}")
    for result in report["results"]:
        print(
            f"{result['name']:38} "
            f"{result['median_ns_per_operation']:16.1f} "
            f"{result['min_ns_per_operation']:16.1f}"
        )
    speedup = report["derived"]["independent_transient_parallel_speedup"]
    print(f"\nindependent transient parallel speedup: {speedup:.2f}x")


def compare_with_baseline(report, baseline, max_regression):
    current_meta = report["metadata"]
    baseline_meta = baseline["metadata"]
    if current_meta["free_threaded"] != baseline_meta["free_threaded"]:
        raise SystemExit("baseline and current results use different CPython build modes")
    if current_meta["configuration"] != baseline_meta["configuration"]:
        raise SystemExit("baseline and current benchmark configurations differ")

    baseline_results = {result["name"]: result for result in baseline["results"]}
    regressions = []
    print("\ncomparison with baseline (current / baseline):")
    for result in report["results"]:
        previous = baseline_results.get(result["name"])
        if previous is None:
            continue
        ratio = (
            result["median_ns_per_operation"]
            / previous["median_ns_per_operation"]
        )
        print(f"  {result['name']:38} {ratio:8.3f}x")
        if max_regression is not None and ratio > max_regression:
            regressions.append((result["name"], ratio))

    if regressions:
        details = ", ".join(f"{name}={ratio:.3f}x" for name, ratio in regressions)
        raise SystemExit(f"benchmark regression limit exceeded: {details}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=positive_int, default=4096)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--scale", type=positive_int, default=1)
    parser.add_argument("--workers", type=positive_int, default=min(8, os.cpu_count() or 2))
    parser.add_argument("--json", type=Path, help="write machine-readable results")
    parser.add_argument("--baseline", type=Path, help="compare with a prior JSON report")
    parser.add_argument(
        "--max-regression",
        type=regression_limit,
        help="fail when any median exceeds this baseline ratio",
    )
    args = parser.parse_args()

    if (
        sysconfig.get_config_var("Py_GIL_DISABLED")
        and getattr(sys, "_is_gil_enabled", lambda: True)()
    ):
        raise SystemExit(
            "free-threaded benchmarks require the GIL to remain disabled"
        )

    results, derived = run_benchmarks(
        args.size, args.repeats, args.scale, args.workers
    )
    report = {
        "schema_version": 1,
        "metadata": metadata(args),
        "coverage": SOURCES,
        "results": results,
        "derived": derived,
    }
    print_results(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")

    if args.baseline:
        compare_with_baseline(
            report,
            json.loads(args.baseline.read_text()),
            args.max_regression,
        )


if __name__ == "__main__":
    main()
