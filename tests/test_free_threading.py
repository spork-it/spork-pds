import gc
import operator
import sysconfig
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

import spork.pds as pds


IS_FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
WRONG_THREAD_ERROR = "Transient objects are confined to their creating thread"
INVALID_TRANSIENT_ERROR = "Transient used after persistent() call"
INVALID_SORTED_TRANSIENT_ERROR = "TransientSortedVector already made persistent"


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


class WorkerSortedValue:
    def __init__(self, worker_id):
        self.worker_id = worker_id


class BlockingSortKey:
    def __init__(self):
        self.barrier = None

    def __call__(self, value):
        if isinstance(value, WorkerSortedValue):
            self.barrier.wait(timeout=10)
            return 10_000 + value.worker_id
        return value


class RecordingSortKey:
    def __init__(self):
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        return value


class BarrierHashKey:
    def __init__(self, value):
        self.value = value
        self.barrier = None

    def __hash__(self):
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        return 20_000 + self.value

    def __eq__(self, other):
        return isinstance(other, BarrierHashKey) and self.value == other.value


class BarrierEqualityKey:
    def __init__(self, value, barrier=None):
        self.value = value
        self.barrier = barrier

    def __hash__(self):
        return 30_000

    def __eq__(self, other):
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        return (
            isinstance(other, BarrierEqualityKey)
            and self.value == other.value
        )


class BarrierSortedValue:
    def __init__(self, value):
        self.value = value


class BarrierSortedKey:
    def __init__(self):
        self.barrier = None

    def __call__(self, value):
        if isinstance(value, BarrierSortedValue):
            if self.barrier is not None:
                self.barrier.wait(timeout=10)
            return value.value
        return value


class CyclePayload:
    pass


def _call_on_other_thread(operations):
    def invoke_all():
        results = []
        for name, operation in operations:
            try:
                operation()
            except Exception as exc:  # Report the exact failure to the owner.
                results.append((name, type(exc).__name__, str(exc)))
            else:
                results.append((name, None, None))
        return results

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(invoke_all).result(timeout=20)


def _assert_wrong_thread_errors(results):
    assert results
    assert results == [
        (name, "RuntimeError", WRONG_THREAD_ERROR) for name, _, _ in results
    ]


@pytest.mark.skipif(not IS_FREE_THREADED, reason="requires free-threaded CPython")
def test_every_transient_entry_point_rejects_the_wrong_thread_before_access():
    vector = pds.Vector(range(1100)).transient()
    vector_iterator = iter(vector)
    mapping = pds.Map((index, index * 2) for index in range(64)).transient()
    set_value = pds.Set(range(64)).transient()
    sort_key = RecordingSortKey()
    sorted_value = pds.SortedVector(range(64), key=sort_key).transient()
    double_vector = pds.DoubleVector(range(64)).transient()
    int_vector = pds.IntVector(range(64)).transient()
    sort_key_calls = sort_key.calls

    vector_operations = [
        ("vector len", lambda: len(vector)),
        ("vector getitem", lambda: vector[0]),
        ("vector setitem", lambda: operator.setitem(vector, 0, -1)),
        ("vector delitem", lambda: operator.delitem(vector, -1)),
        ("vector contains", lambda: 10 in vector),
        ("vector iter", lambda: iter(vector)),
        ("vector iterator next", lambda: next(vector_iterator)),
        ("vector conj_mut", lambda: vector.conj_mut(1100)),
        ("vector assoc_mut", lambda: vector.assoc_mut(0, -1)),
        ("vector pop_mut", lambda: vector.pop_mut()),
        ("vector append", lambda: vector.append(1100)),
        ("vector extend", lambda: vector.extend([1100, 1101])),
        ("vector sort", lambda: vector.sort(reverse=True)),
        ("vector persistent", lambda: vector.persistent()),
    ]
    map_operations = [
        ("map len", lambda: len(mapping)),
        ("map getitem", lambda: mapping[0]),
        ("map setitem", lambda: operator.setitem(mapping, "new", 1)),
        ("map delitem", lambda: operator.delitem(mapping, 0)),
        ("map contains", lambda: 1 in mapping),
        ("map iter", lambda: iter(mapping)),
        ("map assoc_mut", lambda: mapping.assoc_mut("new", 1)),
        ("map dissoc_mut", lambda: mapping.dissoc_mut(0)),
        ("map get", lambda: mapping.get(0)),
        ("map keys", lambda: mapping.keys()),
        ("map values", lambda: mapping.values()),
        ("map items", lambda: mapping.items()),
        ("map persistent", lambda: mapping.persistent()),
    ]
    set_operations = [
        ("set len", lambda: len(set_value)),
        ("set contains", lambda: 1 in set_value),
        ("set iter", lambda: iter(set_value)),
        ("set conj_mut", lambda: set_value.conj_mut(100)),
        ("set disj_mut", lambda: set_value.disj_mut(1)),
        ("set add", lambda: set_value.add(100)),
        ("set discard", lambda: set_value.discard(1)),
        ("set remove", lambda: set_value.remove(1)),
        ("set clear", lambda: set_value.clear()),
        ("set persistent", lambda: set_value.persistent()),
    ]
    sorted_operations = [
        ("sorted vector len", lambda: len(sorted_value)),
        ("sorted vector conj_mut", lambda: sorted_value.conj_mut(100)),
        ("sorted vector disj_mut", lambda: sorted_value.disj_mut(1)),
        ("sorted vector persistent", lambda: sorted_value.persistent()),
    ]
    typed_operations = [
        ("double vector conj_mut", lambda: double_vector.conj_mut(64)),
        ("double vector persistent", lambda: double_vector.persistent()),
        ("int vector conj_mut", lambda: int_vector.conj_mut(64)),
        ("int vector persistent", lambda: int_vector.persistent()),
    ]

    results = _call_on_other_thread(
        vector_operations
        + map_operations
        + set_operations
        + sorted_operations
        + typed_operations
    )
    _assert_wrong_thread_errors(results)

    # No rejected operation may invoke callbacks, advance cursors, or mutate.
    assert sort_key.calls == sort_key_calls
    assert list(vector) == list(range(1100))
    assert next(vector_iterator) == 0
    assert dict(mapping.items()) == {index: index * 2 for index in range(64)}
    assert set(set_value) == set(range(64))
    assert len(sorted_value) == 64

    vector.append(1100)
    mapping.assoc_mut("owner", True)
    set_value.add(100)
    sorted_value.conj_mut(100)
    double_vector.conj_mut(64)
    int_vector.conj_mut(64)

    assert vector.persistent()[-1] == 1100
    assert mapping.persistent()["owner"] is True
    assert 100 in set_value.persistent()
    assert list(sorted_value.persistent())[-1] == 100
    assert double_vector.persistent()[-1] == 64.0
    assert int_vector.persistent()[-1] == 64


@pytest.mark.skipif(not IS_FREE_THREADED, reason="requires free-threaded CPython")
def test_wrong_thread_error_precedes_existing_invalid_transient_errors():
    cases = [
        (
            pds.Vector([1]).transient(),
            lambda transient: transient.append(2),
            INVALID_TRANSIENT_ERROR,
        ),
        (
            pds.Map({"a": 1}).transient(),
            lambda transient: transient.assoc_mut("b", 2),
            INVALID_TRANSIENT_ERROR,
        ),
        (
            pds.Set([1]).transient(),
            lambda transient: transient.add(2),
            INVALID_TRANSIENT_ERROR,
        ),
        (
            pds.SortedVector([1]).transient(),
            lambda transient: transient.conj_mut(2),
            INVALID_SORTED_TRANSIENT_ERROR,
        ),
        (
            pds.DoubleVector([1]).transient(),
            lambda transient: transient.conj_mut(2),
            INVALID_TRANSIENT_ERROR,
        ),
        (
            pds.IntVector([1]).transient(),
            lambda transient: transient.conj_mut(2),
            INVALID_TRANSIENT_ERROR,
        ),
    ]

    for transient, operation, owner_error in cases:
        transient.persistent()
        _assert_wrong_thread_errors(
            _call_on_other_thread([("invalid transient", lambda: operation(transient))])
        )
        with pytest.raises(RuntimeError) as exc_info:
            operation(transient)
        assert str(exc_info.value) == owner_error


def test_parallel_operations_allow_blocking_python_callbacks():
    worker_count = 8
    start_barrier = Barrier(worker_count)

    hash_key = BarrierHashKey(7)
    hash_mapping = pds.Map({hash_key: "hash callback"})
    hash_key.barrier = Barrier(worker_count)

    stored_equal_key = BarrierEqualityKey(9)
    equality_mapping = pds.Map({stored_equal_key: "equality callback"})
    equality_barrier = Barrier(worker_count)

    sorted_key = BarrierSortedKey()
    sorted_value = pds.SortedVector(range(32), key=sorted_key)
    sorted_key.barrier = Barrier(worker_count)

    def exercise(worker_id):
        start_barrier.wait(timeout=10)
        hash_result = hash_mapping[hash_key]
        equality_result = equality_mapping[
            BarrierEqualityKey(9, equality_barrier)
        ]
        item = BarrierSortedValue(100 + worker_id)
        updated = sorted_value.conj(item)
        return (
            hash_result,
            equality_result,
            updated[-1] is item,
            len(updated),
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(exercise, range(worker_count)))

    assert results == [
        ("hash callback", "equality callback", True, len(sorted_value) + 1)
    ] * worker_count

    hash_key.barrier = None
    sorted_key.barrier = None
    assert hash_mapping[hash_key] == "hash callback"
    assert equality_mapping[BarrierEqualityKey(9)] == "equality callback"
    assert list(sorted_value) == list(range(32))


@pytest.mark.skipif(not IS_FREE_THREADED, reason="requires free-threaded CPython")
def test_concurrent_cycle_collection_and_lifecycle_churn():
    worker_count = 6
    iterations = 120
    start_barrier = Barrier(worker_count)

    def churn(worker_id):
        references = []
        start_barrier.wait(timeout=10)
        for iteration in range(iterations):
            payload = CyclePayload()
            case = (worker_id + iteration) % 7
            if case == 0:
                container = pds.cons(payload)
                payload.backref = container
            elif case == 1:
                container = pds.Vector([payload])
                payload.backref = container
            elif case == 2:
                container = pds.Map({iteration: payload})
                payload.backref = container
            elif case == 3:
                container = pds.Set([payload])
                payload.backref = container
            elif case == 4:
                container = pds.SortedVector([payload], key=id)
                payload.backref = container
            elif case == 5:
                container = iter(pds.Vector([payload]))
                payload.backref = container
            else:
                container = pds.Map({iteration: payload}).transient()
                payload.backref = container

            references.append(weakref.ref(payload))
            del container, payload
            if iteration % 10 == 0:
                gc.collect()
        return references

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        references = [
            reference
            for worker_references in executor.map(churn, range(worker_count))
            for reference in worker_references
        ]

    for _ in range(3):
        gc.collect()
    assert len(references) == worker_count * iterations
    assert all(reference() is None for reference in references)


def test_independent_transients_do_not_share_editable_nodes():
    worker_count = 6
    append_count = 64
    start_barrier = Barrier(worker_count)
    sort_key = BlockingSortKey()
    sort_key.barrier = Barrier(worker_count)

    vector = pds.Vector(range(1100))
    mapping = pds.Map(
        [(DenseKey(index), ("dense", index)) for index in range(40)]
        + [(CollisionKey(index), ("collision", index)) for index in range(20)]
    )
    set_value = pds.Set(
        [DenseKey(index) for index in range(40)]
        + [CollisionKey(index) for index in range(20)]
    )
    sorted_value = pds.SortedVector(range(64), key=sort_key)
    double_vector = pds.DoubleVector(range(1100))
    int_vector = pds.IntVector(range(1100))

    def build(worker_id):
        vector_builder = vector.transient()
        map_builder = mapping.transient()
        set_builder = set_value.transient()
        sorted_builder = sorted_value.transient()
        double_builder = double_vector.transient()
        int_builder = int_vector.transient()
        sorted_item = WorkerSortedValue(worker_id)

        start_barrier.wait(timeout=10)
        vector_builder.assoc_mut(1024, ("worker", worker_id))
        map_builder.assoc_mut(CollisionKey(5), ("updated", worker_id))
        map_builder.dissoc_mut(DenseKey(worker_id))
        set_builder.disj_mut(DenseKey(worker_id))
        sorted_builder.disj_mut(worker_id)
        sorted_builder.conj_mut(sorted_item)

        for offset in range(append_count):
            marker = (worker_id, offset)
            vector_builder.conj_mut(marker)
            collision = CollisionKey(1000 + worker_id * append_count + offset)
            map_builder.assoc_mut(collision, marker)
            set_builder.conj_mut(collision)
            double_builder.conj_mut(worker_id * 1000 + offset)
            int_builder.conj_mut(worker_id * 1000 + offset)

        return (
            vector_builder.persistent(),
            map_builder.persistent(),
            set_builder.persistent(),
            sorted_builder.persistent(),
            double_builder.persistent(),
            int_builder.persistent(),
            sorted_item,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(build, range(worker_count)))

    for worker_id, result in enumerate(results):
        (
            built_vector,
            built_map,
            built_set,
            built_sorted,
            built_doubles,
            built_ints,
            sorted_item,
        ) = result
        markers = [(worker_id, offset) for offset in range(append_count)]

        assert built_vector[1024] == ("worker", worker_id)
        assert list(built_vector[-append_count:]) == markers
        assert built_map[CollisionKey(5)] == ("updated", worker_id)
        assert DenseKey(worker_id) not in built_map
        assert len(built_map) == len(mapping) - 1 + append_count
        assert DenseKey(worker_id) not in built_set
        assert len(built_set) == len(set_value) - 1 + append_count
        assert worker_id not in built_sorted
        assert list(built_sorted)[-1] is sorted_item
        assert list(built_doubles[-append_count:]) == [
            float(worker_id * 1000 + offset) for offset in range(append_count)
        ]
        assert list(built_ints[-append_count:]) == [
            worker_id * 1000 + offset for offset in range(append_count)
        ]

    # Every source value is unchanged after all independent builders finish.
    assert vector[1024] == 1024
    assert len(vector) == 1100
    assert mapping[CollisionKey(5)] == ("collision", 5)
    assert all(DenseKey(index) in mapping for index in range(40))
    assert len(mapping) == 60
    assert all(DenseKey(index) in set_value for index in range(40))
    assert len(set_value) == 60
    assert list(sorted_value) == list(range(64))
    assert list(double_vector) == [float(value) for value in range(1100)]
    assert list(int_vector) == list(range(1100))


@pytest.mark.skipif(IS_FREE_THREADED, reason="checks regular-GIL compatibility")
def test_regular_gil_build_preserves_cross_thread_transient_behavior():
    transient = pds.Vector([1, 2, 3]).transient()

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(transient.append, 4).result(timeout=10) is transient

    assert list(transient.persistent()) == [1, 2, 3, 4]
