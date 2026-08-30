"""High-contention semantic stress tests for native free-threading.

This runner intentionally uses only the standard library so it can execute
under sanitizer-instrumented CPython builds without installing pytest.
"""

import argparse
import gc
import os
import sys
import sysconfig
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import spork.pds as pds


class DenseKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return self.value

    def __eq__(self, other):
        return isinstance(other, DenseKey) and self.value == other.value


class CollisionKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 0x5A5A

    def __eq__(self, other):
        return isinstance(other, CollisionKey) and self.value == other.value


class CyclePayload:
    pass


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def make_cons(size):
    value = None
    for item in reversed(range(size)):
        value = pds.cons(item, value)
    return value


def stress_first_publication(workers, rounds, timeout):
    hash_values = []
    buffer_values = []
    for round_id in range(rounds):
        data = range(round_id % 17, round_id % 17 + 64)
        hash_values.append(
            (
                make_cons(32),
                pds.Vector(data),
                pds.Map((item, item * 2) for item in data),
                pds.Set(data),
                pds.DoubleVector(data),
                pds.IntVector(data),
            )
        )
        buffer_values.append((pds.DoubleVector(data), pds.IntVector(data)))

    barrier = Barrier(workers)

    def publish(_worker_id):
        observed = []
        for round_id in range(rounds):
            barrier.wait(timeout=timeout)
            hashes = tuple(hash(value) for value in hash_values[round_id])
            double_vector, int_vector = buffer_values[round_id]
            double_view = memoryview(double_vector)
            int_view = memoryview(int_vector)
            assert double_view.obj is double_vector
            assert int_view.obj is int_vector
            assert double_view.readonly and int_view.readonly
            assert double_view[0] == float(round_id % 17)
            assert int_view[-1] == round_id % 17 + 63
            observed.append(hashes)
        return observed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(publish, range(workers)))

    assert results == [results[0]] * workers
    for values, expected in zip(hash_values, results[0], strict=True):
        assert tuple(hash(value) for value in values) == expected


def stress_typed_histories(workers, steps, timeout):
    double_base = pds.DoubleVector(range(1100))
    int_base = pds.IntVector(range(1100))
    barrier = Barrier(workers)

    def build(worker_id):
        doubles = double_base
        integers = int_base
        double_history = []
        int_history = []
        barrier.wait(timeout=timeout)

        for step in range(steps):
            old_doubles = doubles
            old_integers = integers

            double_builder = doubles.transient()
            int_builder = integers.transient()
            double_builder.conj_mut(worker_id * 1_000_000 + step)
            int_builder.conj_mut(worker_id * 1_000_000 + step)
            doubles = double_builder.persistent()
            integers = int_builder.persistent()

            assert len(old_doubles) == 1100 + step
            assert len(old_integers) == 1100 + step
            if step % 17 == 0:
                double_history.append(old_doubles)
                int_history.append(old_integers)

        for history_index, (old_doubles, old_integers) in enumerate(
            zip(double_history, int_history, strict=True)
        ):
            expected_length = 1100 + history_index * 17
            assert len(old_doubles) == expected_length
            assert len(old_integers) == expected_length
            assert old_doubles[1099] == 1099.0
            assert old_integers[1099] == 1099

        return doubles, integers

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(build, range(workers)))

    for worker_id, (doubles, integers) in enumerate(results):
        assert len(doubles) == 1100 + steps
        assert len(integers) == 1100 + steps
        assert doubles[-1] == float(worker_id * 1_000_000 + steps - 1)
        assert integers[-1] == worker_id * 1_000_000 + steps - 1
    assert list(double_base) == [float(value) for value in range(1100)]
    assert list(int_base) == list(range(1100))


def stress_independent_hamt_transients(workers, steps, timeout):
    dense_keys = [DenseKey(value) for value in range(48)]
    collision_keys = [CollisionKey(value) for value in range(24)]
    base_map = pds.Map(
        [(key, ("dense", key.value)) for key in dense_keys]
        + [(key, ("collision", key.value)) for key in collision_keys]
    )
    base_set = pds.Set(dense_keys + collision_keys)
    barrier = Barrier(workers)

    def mutate(worker_id):
        map_builder = base_map.transient()
        set_builder = base_set.transient()
        barrier.wait(timeout=timeout)

        map_builder.dissoc_mut(DenseKey(worker_id % len(dense_keys)))
        set_builder.disj_mut(DenseKey(worker_id % len(dense_keys)))
        map_builder.assoc_mut(CollisionKey(5), ("updated", worker_id))

        for step in range(steps):
            value = worker_id * steps + step + 10_000
            if step % 2:
                key = DenseKey(value)
            else:
                key = CollisionKey(value)
            map_builder.assoc_mut(key, (worker_id, step))
            set_builder.conj_mut(key)

        return map_builder.persistent(), set_builder.persistent()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(mutate, range(workers)))

    for worker_id, (mapping, set_value) in enumerate(results):
        assert DenseKey(worker_id % len(dense_keys)) not in mapping
        assert DenseKey(worker_id % len(dense_keys)) not in set_value
        assert mapping[CollisionKey(5)] == ("updated", worker_id)
        for step in range(steps):
            value = worker_id * steps + step + 10_000
            key = DenseKey(value) if step % 2 else CollisionKey(value)
            assert mapping[key] == (worker_id, step)
            assert key in set_value

    assert len(base_map) == len(dense_keys) + len(collision_keys)
    assert len(base_set) == len(dense_keys) + len(collision_keys)
    assert base_map[CollisionKey(5)] == ("collision", 5)
    assert DenseKey(0) in base_set


def stress_gc_cycles(workers, iterations, timeout):
    barrier = Barrier(workers)

    def churn(worker_id):
        references = []
        barrier.wait(timeout=timeout)
        for iteration in range(iterations):
            payload = CyclePayload()
            case = (worker_id + iteration) % 7
            if case == 0:
                container = pds.cons(payload)
            elif case == 1:
                container = pds.Vector([payload])
            elif case == 2:
                container = pds.Map({iteration: payload})
            elif case == 3:
                container = pds.Set([payload])
            elif case == 4:
                container = pds.SortedVector([payload], key=id)
            elif case == 5:
                container = iter(pds.Vector([payload]))
            else:
                container = pds.Map({iteration: payload}).transient()
            payload.backref = container
            references.append(weakref.ref(payload))
            del container, payload
            if iteration % 10 == 0:
                gc.collect()
        return references

    with ThreadPoolExecutor(max_workers=workers) as executor:
        references = [
            reference
            for worker_references in executor.map(churn, range(workers))
            for reference in worker_references
        ]

    for _ in range(3):
        gc.collect()
    assert len(references) == workers * iterations
    assert all(reference() is None for reference in references)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=positive_int, default=min(8, os.cpu_count() or 2))
    parser.add_argument("--publication-rounds", type=positive_int, default=1000)
    parser.add_argument("--history-steps", type=positive_int, default=400)
    parser.add_argument("--hamt-steps", type=positive_int, default=300)
    parser.add_argument("--gc-iterations", type=positive_int, default=500)
    parser.add_argument("--timeout", type=positive_int, default=120)
    parser.add_argument(
        "--require-no-gil",
        action="store_true",
        help="fail unless running a free-threaded interpreter with the GIL disabled",
    )
    args = parser.parse_args()

    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    if args.require_no_gil and (not free_threaded or gil_enabled):
        raise SystemExit("a free-threaded CPython process with the GIL disabled is required")

    print(
        f"Python {sys.version.split()[0]} free_threaded={free_threaded} "
        f"gil_enabled={gil_enabled} workers={args.workers}"
    )
    stress_first_publication(args.workers, args.publication_rounds, args.timeout)
    print(
        f"first hash/buffer publication: {args.publication_rounds} rounds, "
        f"{args.publication_rounds * args.workers} synchronized worker starts"
    )
    stress_typed_histories(args.workers, args.history_steps, args.timeout)
    print(f"typed persistent histories: {args.history_steps} steps per worker")
    stress_independent_hamt_transients(args.workers, args.hamt_steps, args.timeout)
    print(f"independent HAMT transients: {args.hamt_steps} mutations per worker")
    stress_gc_cycles(args.workers, args.gc_iterations, args.timeout)
    print(f"concurrent GC-cycle churn: {args.gc_iterations} cycles per worker")
    print("free-threading stress suite passed")


if __name__ == "__main__":
    main()
