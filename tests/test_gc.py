import gc
import sys
import weakref

import pytest

from spork.pds import (
    Cons,
    Map,
    Set,
    SortedVector,
    Vector,
    cons,
    hash_map,
    hash_set,
    sorted_vec,
    vec,
)


class Payload:
    def __hash__(self):
        return 0


class CollisionKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, CollisionKey) and self.value == other.value


class DenseKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return self.value

    def __eq__(self, other):
        return isinstance(other, DenseKey) and self.value == other.value


class KeyFunction:
    def __init__(self, payload):
        self.payload = payload

    def __call__(self, _):
        return 0


def _persistent_cons(payload):
    return Cons(payload)


def _persistent_vector_tail(payload):
    return Vector([payload])


def _persistent_vector_tree(payload):
    return Vector([*range(40), payload])


def _persistent_vector_deep_tree(payload):
    return Vector([*range(1100), payload])


def _persistent_map(payload):
    return Map({"payload": payload})


def _persistent_map_key(payload):
    return Map({payload: "value"})


def _persistent_map_collision(payload):
    return Map([(CollisionKey(1), None), (CollisionKey(2), payload)])


def _persistent_map_dense_node(payload):
    return Map([(DenseKey(index), payload if index == 0 else index) for index in range(20)])


def _persistent_set(payload):
    return Set([payload])


def _persistent_set_collision(payload):
    return Set([CollisionKey(1), CollisionKey(2), payload])


def _persistent_set_dense_node(payload):
    return Set([payload, *(DenseKey(index) for index in range(20))])


def _persistent_sorted_vector(payload):
    return SortedVector([payload], key=lambda _: 0)


def _persistent_sorted_vector_key_function(payload):
    return SortedVector([1], key=KeyFunction(payload))


def _transient_vector(payload):
    value = Vector(range(40)).transient()
    value.append(payload)
    return value


def _transient_map(payload):
    value = Map().transient()
    value["payload"] = payload
    return value


def _transient_set(payload):
    value = Set().transient()
    value.add(payload)
    return value


def _transient_sorted_vector(payload):
    value = SortedVector([], key=lambda _: 0).transient()
    value.conj_mut(payload)
    return value


OBJECT_CONTAINERS = [
    _persistent_cons,
    _persistent_vector_tail,
    _persistent_vector_tree,
    _persistent_vector_deep_tree,
    _persistent_map,
    _persistent_map_key,
    _persistent_map_collision,
    _persistent_map_dense_node,
    _persistent_set,
    _persistent_set_collision,
    _persistent_set_dense_node,
    _persistent_sorted_vector,
    _persistent_sorted_vector_key_function,
    _transient_vector,
    _transient_map,
    _transient_set,
    _transient_sorted_vector,
]


@pytest.mark.parametrize("make_container", OBJECT_CONTAINERS)
def test_object_containers_are_tracked_and_expose_their_reference_graph(
    make_container,
):
    payload = Payload()
    container = make_container(payload)

    assert gc.is_tracked(container)

    pending = [container]
    seen = set()
    while pending:
        current = pending.pop()
        if current is payload:
            break
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        pending.extend(gc.get_referents(current))
    else:
        pytest.fail("payload is missing from the container's GC reference graph")


@pytest.mark.parametrize("make_container", OBJECT_CONTAINERS)
def test_cycles_through_object_containers_are_collected(make_container):
    def make_cycle():
        payload = Payload()
        container = make_container(payload)
        payload.backref = container
        return weakref.ref(payload)

    payload_ref = make_cycle()
    gc.collect()

    assert payload_ref() is None


def test_dense_hamt_packing_does_not_retain_values():
    payload = Payload()
    baseline = sys.getrefcount(payload)
    keys = [DenseKey(index) for index in range(20)]
    value = Map([(key, payload if index == 0 else index) for index, key in enumerate(keys)])

    for index in range(19, 6, -1):
        value = value.dissoc(keys[index])

    assert value[keys[0]] is payload
    del value
    gc.collect()

    assert sys.getrefcount(payload) == baseline


def _failing_values(payload):
    yield payload
    raise RuntimeError("iteration failed")


def _failing_pairs(payload):
    yield "payload", payload
    raise RuntimeError("iteration failed")


@pytest.mark.parametrize(
    "build",
    [
        lambda payload: Vector(_failing_values(payload)),
        lambda payload: Map(_failing_pairs(payload)),
        lambda payload: Set(_failing_values(payload)),
        lambda payload: SortedVector(_failing_values(payload), key=lambda _: 0),
    ],
)
def test_partial_construction_releases_values(build):
    payload = Payload()
    baseline = sys.getrefcount(payload)

    with pytest.raises(RuntimeError, match="iteration failed"):
        build(payload)
    gc.collect()

    assert sys.getrefcount(payload) == baseline


def _iterator_cycle(make_container):
    payload = Payload()
    container = make_container(payload)
    iterator = iter(container)
    payload.backref = iterator
    return weakref.ref(payload), iterator


@pytest.mark.parametrize(
    "make_container",
    [
        _persistent_cons,
        _persistent_vector_tree,
        _persistent_vector_deep_tree,
        _persistent_map,
        _persistent_map_key,
        _persistent_map_collision,
        _persistent_map_dense_node,
        _persistent_set,
        _persistent_set_collision,
        _persistent_set_dense_node,
        _persistent_sorted_vector,
        _persistent_sorted_vector_key_function,
        _transient_vector,
        _transient_map,
        _transient_set,
    ],
)
def test_cycles_through_iterators_are_collected(make_container):
    payload_ref, iterator = _iterator_cycle(make_container)
    assert gc.is_tracked(iterator)

    del iterator
    gc.collect()

    assert payload_ref() is None


@pytest.mark.parametrize(
    "make_value",
    [
        lambda payload: cons(payload),
        lambda payload: vec(payload),
        lambda payload: vec([payload]),
        lambda payload: hash_map("payload", payload),
        lambda payload: hash_set([payload]),
        lambda payload: sorted_vec([payload], key=lambda _: 0),
    ],
)
def test_factories_release_temporary_builder_references(make_value):
    payload = Payload()
    baseline = sys.getrefcount(payload)

    value = make_value(payload)
    assert sys.getrefcount(payload) >= baseline + 1

    del value
    gc.collect()

    assert sys.getrefcount(payload) == baseline
