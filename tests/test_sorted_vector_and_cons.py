import operator
import pickle

import pytest

from spork_pds import Cons, SortedVector, cons, sorted_vec


def test_sorted_vector_ordering_indexing_and_queries():
    value = sorted_vec([3, 1, 2, 2, 5, 4])

    assert isinstance(value, SortedVector)
    assert list(value) == [1, 2, 2, 3, 4, 5]
    assert len(value) == 6
    assert value[0] == 1
    assert value[-1] == 5
    assert value.nth(99, "missing") == "missing"
    assert value.first() == 1
    assert value.last() == 5
    assert value.index_of(3) == 3
    assert value.index_of(99) == -1
    assert value.rank(3) == 3
    assert 4 in value
    assert 99 not in value


def test_sorted_vector_persistent_updates_and_duplicates():
    original = sorted_vec([3, 1, 2])
    added = original.conj(2).conj(4)
    removed = added.disj(2)
    unchanged = original.disj(99)

    assert list(original) == [1, 2, 3]
    assert list(added) == [1, 2, 2, 3, 4]
    assert list(removed) == [1, 2, 3, 4]
    assert unchanged is original


def test_sorted_vector_key_and_reverse_order():
    by_length = sorted_vec(["banana", "fig", "apple", "date"], key=len)
    descending = sorted_vec([1, 4, 2, 3], reverse=True)

    assert [len(item) for item in by_length] == [3, 4, 5, 6]
    assert list(descending) == [4, 3, 2, 1]
    assert list(descending.conj(5)) == [5, 4, 3, 2, 1]


def test_sorted_vector_transient_and_invalidation():
    original = sorted_vec([3, 1])
    transient = original.transient()
    transient.conj_mut(2)
    transient.conj_mut(4)
    transient.disj_mut(1)

    result = transient.persistent()
    assert list(original) == [1, 3]
    assert list(result) == [2, 3, 4]

    with pytest.raises(RuntimeError):
        transient.conj_mut(5)


def test_sorted_vector_equality_hash_and_pickle():
    left = sorted_vec([3, 1, 2])
    right = sorted_vec([1, 2, 3])

    assert left == right
    assert hash(left) == hash(right)
    assert pickle.loads(pickle.dumps(left)) == left

    keyed = sorted_vec([1, 2, 3], key=operator.neg, reverse=True)
    restored = pickle.loads(pickle.dumps(keyed))
    assert list(restored) == list(keyed)
    assert list(restored.conj(4)) == list(keyed.conj(4))


def test_cons_properties_iteration_and_prepend():
    tail = cons(2, cons(3))
    value = cons(1, tail)
    prepended = value.conj(0)

    assert isinstance(value, Cons)
    assert value.first == 1
    assert value.rest is tail
    assert list(value) == [1, 2, 3]
    assert len(value) == 3
    assert list(prepended) == [0, 1, 2, 3]
    assert list(value) == [1, 2, 3]
    assert repr(value) == "(1 2 3)"


def test_cons_equality_hash_and_pickle():
    left = cons(1, cons(2, cons(3)))
    right = cons(1, cons(2, cons(3)))

    assert left == right
    assert hash(left) == hash(right)
    assert pickle.loads(pickle.dumps(left)) == left
