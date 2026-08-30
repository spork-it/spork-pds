import operator
import pickle

import pytest

from spork.pds import Cons, SortedVector, cons, sorted_vec


def test_sorted_vector_ordering_indexing_and_queries():
    value = sorted_vec([3, 1, 2, 2, 5, 4])
    default = object()

    assert isinstance(value, SortedVector)
    assert list(value) == [1, 2, 2, 3, 4, 5]
    assert len(value) == 6
    assert value[0] == 1
    assert value[-1] == 5
    assert value.nth(99, default) is default
    assert value.first() == 1
    assert value.last() == 5
    assert value.index_of(2) == 1
    assert value.index_of(3) == 3
    assert value.index_of(99) == -1
    assert value.rank(0) == 0
    assert value.rank(2) == 1
    assert value.rank(3) == 3
    assert value.rank(99) == len(value)
    assert 4 in value
    assert 99 not in value

    with pytest.raises(IndexError):
        value[99]
    with pytest.raises(IndexError):
        value.nth(-99)
    with pytest.raises(TypeError):
        value[1:3]

    empty = sorted_vec()
    assert empty.first() is None
    assert empty.last() is None
    assert empty.nth(0, default) is default
    assert repr(empty) == "sorted_vec()"


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
    assert descending.rank(3) == 1
    assert descending.rank(5) == 0
    assert descending.rank(0) == 4


def test_sorted_vector_equal_keys_use_exact_values_for_queries_and_removal():
    values = ["a", "b", "c", "d", "ee", "f", "g", "h", "i"]
    value = sorted_vec(values, key=len)

    assert list(value) == ["a", "b", "c", "d", "f", "g", "h", "i", "ee"]
    assert value.index_of("z") == -1
    assert "z" not in value

    for item in values:
        removed = value.disj(item)
        expected = list(value)
        expected.remove(item)
        assert list(removed) == expected
        assert item not in removed
        assert list(value) == ["a", "b", "c", "d", "f", "g", "h", "i", "ee"]


def test_sorted_vector_duplicate_deletion_retains_one_fewer_value():
    value = sorted_vec([index // 5 for index in range(250)])

    while 20 in value:
        before = list(value)
        value = value.disj(20)
        before.remove(20)
        assert list(value) == before
        assert len(value) == len(before)


def test_sorted_vector_transient_and_invalidation():
    original = sorted_vec(["bbb", "a"], key=len, reverse=True)
    transient = original.transient()
    assert transient.conj_mut("cc") is transient
    assert transient.conj_mut("dddd") is transient
    assert transient.disj_mut("a") is transient
    assert transient.disj_mut("missing") is transient

    result = transient.persistent()
    assert list(original) == ["bbb", "a"]
    assert list(result) == ["dddd", "bbb", "cc"]

    for operation in (
        lambda: len(transient),
        lambda: transient.conj_mut("x"),
        lambda: transient.disj_mut("cc"),
        lambda: transient.persistent(),
    ):
        with pytest.raises(RuntimeError):
            operation()


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
    assert value.first == value._first == 1
    assert value.rest is value._rest is tail
    assert list(value) == [1, 2, 3]
    assert len(value) == 3
    assert list(prepended) == [0, 1, 2, 3]
    assert list(value) == [1, 2, 3]
    assert repr(value) == "(1 2 3)"

    with pytest.raises(AttributeError):
        value.first = 99
    with pytest.raises(AttributeError):
        value.rest = None
    with pytest.raises(TypeError):
        cons()
    with pytest.raises(TypeError):
        cons(1, None, None)


def test_cons_cannot_be_reinitialized():
    value = Cons(1, None)

    with pytest.raises(TypeError, match="cannot be reinitialized"):
        Cons.__init__(value, 2, None)

    assert value.first == 1
    assert value.rest is None


def test_cons_comparison_and_hash_errors():
    value = cons(1, cons(2))

    assert value != cons(1)
    assert value != cons(1, cons(3))
    assert value != [1, 2]
    with pytest.raises(TypeError):
        value < cons(1, cons(3))
    with pytest.raises(TypeError):
        hash(cons([]))


def test_cons_equality_hash_and_pickle():
    left = cons(1, cons(2, cons(3)))
    right = cons(1, cons(2, cons(3)))

    assert left == right
    assert hash(left) == hash(right)
    assert pickle.loads(pickle.dumps(left)) == left
