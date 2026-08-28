import pickle

import pytest

from spork.pds import Set, hash_set


class CollisionValue:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, CollisionValue) and self.value == other.value


def test_set_factory_and_persistent_operations():
    original = hash_set([1, 2, 3, 2])
    added = original.conj(4)
    removed = added.disj(2)

    assert isinstance(original, Set)
    assert len(original) == 3
    assert set(original) == {1, 2, 3}
    assert set(added) == {1, 2, 3, 4}
    assert set(removed) == {1, 3, 4}
    assert original.conj(2) is original
    assert original.disj(99) is original
    assert original.copy() is original
    assert original.isdisjoint([4, 5])
    assert not original.isdisjoint([3, 4])


def test_set_constructor_accepts_one_iterable():
    assert set(Set()) == set()
    assert set(Set([1, 2, 2])) == {1, 2}
    assert set(Set(iter(range(3)))) == {0, 1, 2}
    assert set(hash_set("aba")) == {"a", "b"}

    with pytest.raises(TypeError):
        Set(1)
    with pytest.raises(TypeError):
        Set([1], [2])
    with pytest.raises(TypeError):
        Set(values=[1, 2])
    with pytest.raises(TypeError):
        hash_set(1)
    with pytest.raises(TypeError):
        hash_set([1], [2])

    value = Set([1, 2])
    with pytest.raises(TypeError):
        value.__init__([3])
    assert set(value) == {1, 2}


def test_set_operators_and_comparisons():
    left = hash_set([1, 2, 3])
    right = hash_set([3, 4])
    subset = hash_set([1, 2])

    assert set(left | right) == {1, 2, 3, 4}
    assert set(left & right) == {3}
    assert set(left - right) == {1, 2}
    assert set(left ^ right) == {1, 2, 4}
    assert set(left | {4, 5}) == {1, 2, 3, 4, 5}
    with pytest.raises(TypeError):
        left | [4, 5]

    reflected_union = {3, 4} | left
    reflected_frozen_union = frozenset({3, 4}) | left
    reflected_intersection = {2, 3, 4} & left
    reflected_difference = {2, 3, 4} - left
    reflected_xor = {2, 3, 4} ^ left
    assert isinstance(reflected_union, Set)
    assert set(reflected_union) == {1, 2, 3, 4}
    assert isinstance(reflected_frozen_union, Set)
    assert set(reflected_frozen_union) == {1, 2, 3, 4}
    assert isinstance(reflected_intersection, Set)
    assert set(reflected_intersection) == {2, 3}
    assert isinstance(reflected_difference, Set)
    assert set(reflected_difference) == {4}
    assert isinstance(reflected_xor, Set)
    assert set(reflected_xor) == {1, 4}

    union_rebound = left
    union_rebound |= {4}
    intersection_rebound = left
    intersection_rebound &= {2, 3, 4}
    difference_rebound = left
    difference_rebound -= {2}
    xor_rebound = left
    xor_rebound ^= {3, 4}

    assert set(union_rebound) == {1, 2, 3, 4}
    assert set(intersection_rebound) == {2, 3}
    assert set(difference_rebound) == {1, 3}
    assert set(xor_rebound) == {1, 2, 4}
    assert all(
        isinstance(result, Set)
        for result in (
            union_rebound,
            intersection_rebound,
            difference_rebound,
            xor_rebound,
        )
    )
    assert set(left) == {1, 2, 3}

    assert subset < left
    assert subset <= left
    assert left > subset
    assert left >= subset
    assert left == hash_set([3, 2, 1])
    assert hash(left) == hash(hash_set([3, 2, 1]))


def test_set_hamt_boundaries_collisions_and_sequence_conversion():
    original = hash_set(range(200))
    value = original
    for item in range(200):
        value = value.disj(item)

    assert len(value) == 0
    assert set(original) == set(range(200))
    assert set(original.to_seq()) == set(range(200))
    assert hash_set().to_seq() is None
    assert repr(hash_set()) == "#{}"

    collisions = hash_set(CollisionValue(index) for index in range(30))
    updated = collisions.disj(CollisionValue(5)).conj(CollisionValue(30))
    assert {item.value for item in collisions} == set(range(30))
    assert {item.value for item in updated} == set(range(30)) - {5} | {30}


def test_set_operators_reject_non_set_iterables():
    value = hash_set([1, 2, 3])

    for operation in (
        lambda: value | [3, 4],
        lambda: value & [3, 4],
        lambda: value - [3, 4],
        lambda: value ^ [3, 4],
    ):
        with pytest.raises(TypeError):
            operation()


def test_transient_set_batch_workflow_and_invalidation():
    original = hash_set([1, 2, 3])
    transient = original.transient()

    assert transient.add(4) is None
    assert transient.conj_mut(5) is transient
    assert transient.discard(2) is None
    assert transient.disj_mut(3) is transient
    assert transient.remove(1) is None

    with pytest.raises(KeyError):
        transient.remove(99)

    assert set(transient) == {4, 5}
    assert 4 in transient
    result = transient.persistent()
    assert set(original) == {1, 2, 3}
    assert set(result) == {4, 5}

    operations = [
        lambda: len(transient),
        lambda: 4 in transient,
        lambda: iter(transient),
        lambda: transient.add(6),
        lambda: transient.conj_mut(6),
        lambda: transient.discard(4),
        lambda: transient.disj_mut(4),
        lambda: transient.remove(4),
        lambda: transient.clear(),
        lambda: transient.persistent(),
    ]
    for operation in operations:
        with pytest.raises(RuntimeError):
            operation()


def test_transient_set_clear():
    transient = hash_set(range(100)).transient()
    transient.clear()
    assert len(transient) == 0
    assert len(transient.persistent()) == 0


def test_set_pickle_round_trip():
    value = hash_set(range(100))
    restored = pickle.loads(pickle.dumps(value))

    assert isinstance(restored, Set)
    assert restored == value
