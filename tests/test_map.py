import gc
import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import MappingProxyType

import pytest

from spork.pds import Map, Vector, hash_map


class CollisionKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, CollisionKey) and self.value == other.value


class PairMapping:
    def __init__(self, pair):
        self.pair = pair

    def items(self):
        return iter((self.pair,))


class PairValue:
    pass


class BlockingHashKey:
    def __init__(self, started, resume):
        self.started = started
        self.resume = resume
        self.block_once = True

    def __hash__(self):
        if self.block_once:
            self.block_once = False
            self.started.set()
            if not self.resume.wait(timeout=10):
                raise AssertionError("pair mutation did not resume hashing")
        return 12345

    def __eq__(self, other):
        return self is other


class FailingPair:
    def __init__(self, first):
        self.first = first

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index == 0:
            return self.first
        self.first = None
        raise RuntimeError("pair access failed")


def test_map_factory_and_mapping_protocols():
    value = hash_map("a", 1, "b", 2)

    assert isinstance(value, Map)
    assert len(value) == 2
    assert value["a"] == 1
    assert value.get("missing") is None
    assert value.get("missing", 99) == 99
    assert "b" in value
    assert set(value.keys()) == {"a", "b"}
    assert set(value.values()) == {1, 2}
    assert dict(value.items()) == {"a": 1, "b": 2}
    assert set(iter(value)) == {"a", "b"}

    with pytest.raises(ValueError):
        hash_map("unpaired")
    with pytest.raises(KeyError):
        value["missing"]


def test_map_constructor_accepts_mappings_pairs_and_keywords():
    assert dict(Map().items()) == {}
    assert dict(Map({"a": 1}, b=2).items()) == {"a": 1, "b": 2}
    assert dict(Map([("a", 1), ("b", 2)]).items()) == {"a": 1, "b": 2}
    assert dict(Map(a=1).items()) == {"a": 1}

    with pytest.raises(TypeError):
        Map({}, {})
    with pytest.raises(TypeError):
        Map(42)
    with pytest.raises(ValueError):
        Map([("not-a-pair",)])
    with pytest.raises(TypeError):
        Map({"a": 1}) | [("b", 2)]

    value = Map({"a": 1})
    with pytest.raises(TypeError):
        value.__init__({"a": 2})
    assert value["a"] == 1


def test_map_persistent_operations_and_merge():
    original = hash_map(*sum(([str(i), i] for i in range(100)), []))
    updated = original.assoc("50", 500).assoc("new", 101)
    removed = updated.dissoc("0")
    unchanged = original.dissoc("missing")
    merged = original | {"50": -1, "extra": 200}
    proxy_merged = original | MappingProxyType({"50": -2, "proxy": 201})
    reflected = {"50": -2, "left": 201} | original

    rebound = original
    rebound |= {"50": -3, "augmented": 202}

    assert len(original) == 100
    assert original["50"] == 50
    assert updated["50"] == 500
    assert "new" in updated and "new" not in original
    assert "0" in original and "0" not in removed
    assert unchanged is original
    assert isinstance(merged, Map)
    assert merged["50"] == -1
    assert merged["extra"] == 200
    assert isinstance(proxy_merged, Map)
    assert proxy_merged["50"] == -2
    assert proxy_merged["proxy"] == 201
    assert isinstance(reflected, Map)
    assert reflected["50"] == 50  # Right-hand Map wins.
    assert reflected["left"] == 201
    assert isinstance(rebound, Map)
    assert rebound["50"] == -3
    assert rebound["augmented"] == 202
    assert rebound is not original
    assert "augmented" not in original
    assert original.copy() is original


@pytest.mark.parametrize(
    "make_pair",
    [
        lambda: ("pair", "tuple"),
        lambda: ["pair", "list"],
        lambda: Vector(["pair", "vector"]),
    ],
)
def test_map_merge_accepts_supported_pair_sequences(make_pair):
    pair = make_pair()
    result = Map() | PairMapping(pair)
    assert dict(result.items()) == {"pair": pair[1]}


def test_map_merge_holds_strong_references_to_mutable_list_pairs():
    started = Event()
    resume = Event()
    key = BlockingHashKey(started, resume)
    value = PairValue()
    value_ref = weakref.ref(value)
    pair = [key, value]
    del value

    retained = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: Map() | PairMapping(pair))
        try:
            assert started.wait(timeout=10)
            pair[:] = ["replacement", object()]
            gc.collect()
            retained = value_ref()
        finally:
            resume.set()
        result = future.result(timeout=10)

    assert retained is not None
    assert result[key] is retained
    assert dict(result.items()) == {key: retained}


def test_map_pair_access_failure_releases_partial_strong_references():
    value = PairValue()
    value_ref = weakref.ref(value)
    pair = FailingPair(value)
    del value

    with pytest.raises(RuntimeError, match="pair access failed"):
        Map() | PairMapping(pair)

    gc.collect()
    assert value_ref() is None


def test_map_handles_hash_collisions():
    keys = [CollisionKey(i) for i in range(20)]
    value = hash_map(*sum(([key, key.value] for key in keys), []))

    assert len(value) == 20
    assert [value[CollisionKey(i)] for i in range(20)] == list(range(20))

    updated = value.assoc(CollisionKey(10), "ten")
    removed = updated.dissoc(CollisionKey(5))
    assert value[CollisionKey(10)] == 10
    assert updated[CollisionKey(10)] == "ten"
    assert CollisionKey(5) not in removed

    transient = value.transient()
    for index in range(0, 20, 2):
        transient.dissoc_mut(CollisionKey(index))
    for index in range(20, 30):
        transient.assoc_mut(CollisionKey(index), index)
    result = transient.persistent()

    assert set(key.value for key in result) == set(range(1, 20, 2)) | set(
        range(20, 30)
    )
    assert len(value) == 20


def test_map_hamt_expansion_and_contraction_preserve_snapshots():
    original = Map({index: str(index) for index in range(200)})
    value = original

    for index in range(0, 200, 2):
        value = value.dissoc(index)
    midpoint = value
    for index in range(1, 200, 2):
        value = value.dissoc(index)

    assert len(value) == 0
    assert dict(original.items()) == {index: str(index) for index in range(200)}
    assert dict(midpoint.items()) == {
        index: str(index) for index in range(1, 200, 2)
    }


def test_map_none_values_errors_and_sequence_conversion():
    value = Map({"present": None, "other": 2})
    default = object()

    assert value["present"] is None
    assert value.get("present", default) is None
    assert value.get("missing", default) is default
    assert {tuple(pair) for pair in value.to_seq()} == {
        ("present", None),
        ("other", 2),
    }
    assert hash_map().to_seq() is None

    with pytest.raises(TypeError):
        value.assoc([], 1)
    with pytest.raises(TypeError):
        [] in value
    with pytest.raises(TypeError):
        hash(Map({"unhashable-value": []}))


def test_transient_map_batch_workflow_and_invalidation():
    original = hash_map("a", 1)
    transient = original.transient()

    assert transient.assoc_mut("b", 2) is transient
    transient["c"] = 3
    assert transient.dissoc_mut("a") is transient
    assert transient.dissoc_mut("missing") is transient
    del transient["b"]

    assert len(transient) == 1
    assert transient["c"] == 3
    assert transient.get("c") == 3
    assert transient.get("missing", 99) == 99
    assert "c" in transient
    assert list(transient.keys()) == ["c"]
    assert list(transient.values()) == [3]
    assert list(transient.items()) == [("c", 3)]
    with pytest.raises(KeyError):
        del transient["missing"]
    with pytest.raises(KeyError):
        transient["missing"]

    result = transient.persistent()
    assert dict(original.items()) == {"a": 1}
    assert dict(result.items()) == {"c": 3}

    operations = [
        lambda: len(transient),
        lambda: transient["c"],
        lambda: "c" in transient,
        lambda: iter(transient),
        lambda: transient.get("c"),
        lambda: transient.keys(),
        lambda: transient.values(),
        lambda: transient.items(),
        lambda: transient.assoc_mut("d", 4),
        lambda: transient.dissoc_mut("c"),
        lambda: transient.persistent(),
    ]
    for operation in operations:
        with pytest.raises(RuntimeError):
            operation()


def test_map_equality_hash_and_pickle():
    left = hash_map("a", 1, "b", 2)
    right = hash_map("b", 2, "a", 1)

    assert left == right
    assert hash(left) == hash(right)

    restored = pickle.loads(pickle.dumps(left))
    assert isinstance(restored, Map)
    assert restored == left
    assert type(restored).__module__ == "spork_pds"
