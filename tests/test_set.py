import pickle

import pytest

from spork_pds import Set, hash_set


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

    with pytest.raises(TypeError):
        Set(1)
    with pytest.raises(TypeError):
        Set([1], [2])
    with pytest.raises(TypeError):
        Set(values=[1, 2])

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

    rebound = left
    rebound -= {2}
    assert isinstance(rebound, Set)
    assert set(rebound) == {1, 3}
    assert rebound is not left
    assert set(left) == {1, 2, 3}

    assert subset < left
    assert subset <= left
    assert left > subset
    assert left >= subset
    assert left == hash_set([3, 2, 1])
    assert hash(left) == hash(hash_set([3, 2, 1]))


def test_transient_set_batch_workflow_and_invalidation():
    original = hash_set([1, 2, 3])
    transient = original.transient()

    transient.add(4)
    transient.conj_mut(5)
    transient.discard(2)
    transient.disj_mut(3)
    transient.remove(1)

    with pytest.raises(KeyError):
        transient.remove(99)

    assert set(transient) == {4, 5}
    result = transient.persistent()
    assert set(original) == {1, 2, 3}
    assert set(result) == {4, 5}

    with pytest.raises(RuntimeError):
        transient.add(6)
    with pytest.raises(RuntimeError):
        list(transient)


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
