import gc
import sysconfig
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

import spork.pds as pds


IS_FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
ITERATOR_BUSY_ERROR = "Iterator is already executing"
WRONG_THREAD_ERROR = "Transient objects are confined to their creating thread"


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
        return 10_000

    def __eq__(self, other):
        return isinstance(other, CollisionKey) and self.value == other.value


def _cons_from(values):
    result = None
    for value in reversed(values):
        result = pds.cons(value, result)
    return result


def _consume_shared(iterator, barrier):
    values = []
    busy_errors = 0
    barrier.wait(timeout=10)

    while True:
        try:
            value = next(iterator)
        except StopIteration:
            return values, busy_errors
        except RuntimeError as exc:
            if str(exc) != ITERATOR_BUSY_ERROR:
                raise
            busy_errors += 1
            if busy_errors > 100_000:
                raise AssertionError("iterator remained busy")
        else:
            values.append(value)


def _submit_shared_consumers(executor, iterator, worker_count):
    barrier = Barrier(worker_count)
    return [
        executor.submit(_consume_shared, iterator, barrier)
        for _ in range(worker_count)
    ]


def _combine_consumer_results(results):
    values = [value for worker_values, _ in results for value in worker_values]
    busy_errors = sum(count for _, count in results)
    return values, busy_errors


def _drain_concurrently(iterator, worker_count=8):
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = _submit_shared_consumers(executor, iterator, worker_count)
        results = [future.result(timeout=30) for future in futures]
    return _combine_consumer_results(results)


def _nested_iterators(iterator):
    pending = [iterator]
    seen = {id(iterator)}
    while pending:
        current = pending.pop()
        for referent in gc.get_referents(current):
            referent_type = type(referent)
            if (
                id(referent) not in seen
                and referent_type.__module__ == "spork_pds"
                and referent_type.__name__.endswith("Iterator")
            ):
                seen.add(id(referent))
                pending.append(referent)
                yield referent


def _shared_iterator_cases():
    vector_values = list(range(4097))
    cons_values = list(range(512))
    sorted_values = list(range(2048))
    double_values = [float(value) for value in range(4097)]
    int_values = list(range(4097))

    dense_keys = [DenseKey(value) for value in range(96)]
    collision_keys = [CollisionKey(value) for value in range(64)]
    map_pairs = [
        (key, ("dense", key.value)) for key in dense_keys
    ] + [(key, ("collision", key.value)) for key in collision_keys]
    mapping = pds.Map(map_pairs)
    bitmap_pairs = [(value, -value) for value in range(8)]
    bitmap_mapping = pds.Map(bitmap_pairs)
    set_items = dense_keys + collision_keys
    set_value = pds.Set(set_items)

    return [
        ("vector", iter(pds.Vector(vector_values)), vector_values),
        (
            "bitmap map iterator",
            iter(bitmap_mapping),
            [key for key, _ in bitmap_pairs],
        ),
        ("map iterator", iter(mapping), [key for key, _ in map_pairs]),
        ("map keys", mapping.keys(), [key for key, _ in map_pairs]),
        ("map values", mapping.values(), [value for _, value in map_pairs]),
        ("map items", mapping.items(), map_pairs),
        ("set", iter(set_value), set_items),
        (
            "sorted vector",
            iter(pds.SortedVector(reversed(sorted_values))),
            sorted_values,
        ),
        ("cons", iter(_cons_from(cons_values)), cons_values),
        (
            "double vector",
            iter(pds.DoubleVector(double_values)),
            double_values,
        ),
        ("int vector", iter(pds.IntVector(int_values)), int_values),
    ]


def test_shared_iterators_never_duplicate_or_lose_elements():
    for name, iterator, expected in _shared_iterator_cases():
        observed, _ = _drain_concurrently(iterator)

        assert Counter(observed) == Counter(expected), name
        assert len(observed) == len(expected), name
        assert list(iterator) == []


def test_nested_hash_collision_iterator_is_safe_when_shared_directly():
    dense_keys = [DenseKey(value) for value in range(96)]
    collision_keys = [CollisionKey(value) for value in range(64)]
    mapping = pds.Map(
        (key, key.value) for key in dense_keys + collision_keys
    )
    iterator = iter(mapping)
    collision_iterator = None
    yielded = None

    for _ in range(len(mapping)):
        yielded = next(iterator)
        collision_iterator = next(
            (
                nested
                for nested in _nested_iterators(iterator)
                if type(nested).__name__ == "HashCollisionNodeIterator"
            ),
            None,
        )
        if collision_iterator is not None:
            break

    assert isinstance(yielded, CollisionKey)
    assert collision_iterator is not None

    observed, _ = _drain_concurrently(collision_iterator)
    expected = [key for key in collision_keys if key != yielded]
    assert Counter(observed) == Counter(expected)
    assert len(observed) == len(expected)


def test_reentrant_next_is_rejected_without_advancing_twice():
    errors = []
    iterator = None

    class ReentrantCons(pds.Cons):
        def __del__(self):
            try:
                next(iterator)
            except Exception as exc:
                errors.append((type(exc).__name__, str(exc)))

    value = ReentrantCons(1, pds.cons(2))
    iterator = iter(value)
    del value

    assert next(iterator) == 1
    assert errors == [("RuntimeError", ITERATOR_BUSY_ERROR)]
    assert list(iterator) == [2]


def _small_iterator_cases():
    map_pairs = [(value, value * 2) for value in range(65)]
    set_values = list(range(65))
    sorted_values = list(range(65))
    cons_values = list(range(65))

    return [
        (iter(pds.Vector(range(129))), list(range(129))),
        (pds.Map(map_pairs).items(), map_pairs),
        (iter(pds.Set(set_values)), set_values),
        (iter(pds.SortedVector(reversed(sorted_values))), sorted_values),
        (iter(_cons_from(cons_values)), cons_values),
        (iter(pds.DoubleVector(range(129))), [float(i) for i in range(129)]),
        (iter(pds.IntVector(range(129))), list(range(129))),
    ]


def test_concurrent_iterator_exhaustion_and_release_stress():
    with ThreadPoolExecutor(max_workers=4) as executor:
        for _ in range(10):
            cases = _small_iterator_cases()
            while cases:
                iterator, expected = cases.pop()
                futures = _submit_shared_consumers(executor, iterator, 4)
                del iterator  # Worker calls now own all iterator references.
                results = [future.result(timeout=30) for future in futures]
                observed, _ = _combine_consumer_results(results)
                assert Counter(observed) == Counter(expected)
            gc.collect()


@pytest.mark.skipif(not IS_FREE_THREADED, reason="requires free-threaded CPython")
def test_transient_iterator_cursors_cannot_bypass_thread_confinement():
    dense_keys = [DenseKey(value) for value in range(96)]
    collision_keys = [CollisionKey(value) for value in range(64)]
    map_pairs = [(key, key.value) for key in dense_keys + collision_keys]
    mapping = pds.Map(map_pairs).transient()
    set_items = dense_keys + collision_keys

    cases = [
        ("vector", iter(pds.Vector(range(257)).transient()), list(range(257))),
        ("map keys", iter(mapping), [key for key, _ in map_pairs]),
        ("map values", mapping.values(), [value for _, value in map_pairs]),
        ("map items", mapping.items(), map_pairs),
        ("set", iter(pds.Set(set_items).transient()), set_items),
        ("empty map", iter(pds.Map().transient()), []),
        ("empty set", iter(pds.Set().transient()), []),
    ]
    barrier = Barrier(8)

    def advance_from_wrong_thread():
        barrier.wait(timeout=10)
        results = []
        for name, iterator, _ in cases:
            try:
                next(iterator)
            except Exception as exc:
                results.append((name, type(exc).__name__, str(exc)))
            else:
                results.append((name, None, None))
        return results

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: advance_from_wrong_thread(), range(8)))

    expected_errors = [
        (name, "RuntimeError", WRONG_THREAD_ERROR) for name, _, _ in cases
    ]
    assert results == [expected_errors] * 8
    for name, iterator, expected in cases:
        assert Counter(iterator) == Counter(expected), name


@pytest.mark.skipif(not IS_FREE_THREADED, reason="requires free-threaded CPython")
def test_nested_hamt_iterators_inherit_transient_thread_confinement():
    dense_keys = [DenseKey(value) for value in range(96)]
    collision_keys = [CollisionKey(value) for value in range(64)]
    mapping = pds.Map(
        (key, key.value) for key in dense_keys + collision_keys
    ).transient()
    iterator = iter(mapping)
    collision_iterator = None

    for _ in range(len(mapping)):
        next(iterator)
        collision_iterator = next(
            (
                nested
                for nested in _nested_iterators(iterator)
                if type(nested).__name__ == "HashCollisionNodeIterator"
            ),
            None,
        )
        if collision_iterator is not None:
            break

    assert collision_iterator is not None

    def advance_from_wrong_thread():
        try:
            next(collision_iterator)
        except Exception as exc:
            return type(exc).__name__, str(exc)
        return None

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(advance_from_wrong_thread).result(timeout=10)

    assert result == ("RuntimeError", WRONG_THREAD_ERROR)
