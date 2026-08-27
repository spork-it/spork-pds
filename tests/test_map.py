import pickle

import pytest

from spork_pds import Map, hash_map


class CollisionKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, CollisionKey) and self.value == other.value


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


def test_map_persistent_operations_and_merge():
    original = hash_map(*sum(([str(i), i] for i in range(100)), []))
    updated = original.assoc("50", 500).assoc("new", 101)
    removed = updated.dissoc("0")
    unchanged = original.dissoc("missing")
    merged = original | {"50": -1, "extra": 200}

    assert len(original) == 100
    assert original["50"] == 50
    assert updated["50"] == 500
    assert "new" in updated and "new" not in original
    assert "0" in original and "0" not in removed
    assert unchanged is original
    assert merged["50"] == -1
    assert merged["extra"] == 200
    assert original.copy() is original


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


def test_transient_map_batch_workflow_and_invalidation():
    original = hash_map("a", 1)
    transient = original.transient()

    transient.assoc_mut("b", 2)
    transient["c"] = 3
    transient.dissoc_mut("a")
    del transient["b"]

    assert len(transient) == 1
    assert transient.get("c") == 3
    assert list(transient.keys()) == ["c"]

    result = transient.persistent()
    assert dict(original.items()) == {"a": 1}
    assert dict(result.items()) == {"c": 3}

    with pytest.raises(RuntimeError):
        transient.assoc_mut("d", 4)
    with pytest.raises(RuntimeError):
        list(transient)


def test_map_equality_hash_and_pickle():
    left = hash_map("a", 1, "b", 2)
    right = hash_map("b", 2, "a", 1)

    assert left == right
    assert hash(left) == hash(right)

    restored = pickle.loads(pickle.dumps(left))
    assert isinstance(restored, Map)
    assert restored == left
    assert type(restored).__module__ == "spork_pds"
