import pickle

import pytest

from spork_pds import EMPTY_VECTOR, Vector, vec


def test_vector_factories_and_sequence_protocols():
    assert list(vec(1, 2, 3)) == [1, 2, 3]
    assert list(vec([1, 2, 3])) == [1, 2, 3]
    assert list(vec(iter(range(3)))) == [0, 1, 2]
    assert list(vec("abc")) == ["abc"]
    assert list(Vector(range(4))) == [0, 1, 2, 3]
    assert list(Vector("ab")) == ["a", "b"]
    with pytest.raises(TypeError):
        Vector(1)
    with pytest.raises(TypeError):
        Vector([1], [2])
    with pytest.raises(TypeError):
        Vector(values=[1, 2])

    value = vec(*range(100))
    assert len(value) == 100
    assert value[0] == 0
    assert value[-1] == 99
    assert list(value[10:20:2]) == [10, 12, 14, 16, 18]
    assert 50 in value

    with pytest.raises(TypeError):
        value.__init__(["replacement"])
    assert list(value) == list(range(100))


def test_vector_persistent_operations_preserve_old_versions():
    original = vec(*range(80))
    appended = original.conj(80)
    updated = original.assoc(40, "changed")
    assoc_at_end = original.assoc(len(original), 80)
    popped = appended.pop()

    assert list(original) == list(range(80))
    assert appended[-1] == 80
    assert updated[40] == "changed"
    assert original[40] == 40
    assert list(assoc_at_end) == list(appended)
    assert list(popped) == list(original)
    assert list(vec(1, 2) + vec(3, 4)) == [1, 2, 3, 4]


def test_vector_operators_are_persistent_and_pythonic():
    original = Vector([1, 2])
    concatenated = original + [3, 4]
    repeated = original * 3
    reflected_repeat = 2 * original

    rebound = original
    rebound += [3]
    multiplied = original
    multiplied *= 2

    assert isinstance(concatenated, Vector)
    assert list(concatenated) == [1, 2, 3, 4]
    assert list(repeated) == [1, 2, 1, 2, 1, 2]
    assert list(reflected_repeat) == [1, 2, 1, 2]
    assert original * 1 is original
    assert original * 0 is EMPTY_VECTOR
    assert original * -1 is EMPTY_VECTOR
    assert list(original) == [1, 2]
    assert list(rebound) == [1, 2, 3]
    assert rebound is not original
    assert list(multiplied) == [1, 2, 1, 2]
    assert multiplied is not original

    with pytest.raises(TypeError):
        [0] + original
    with pytest.raises(TypeError):
        original + 3
    with pytest.raises(TypeError):
        original * 1.5
    with pytest.raises(OverflowError):
        original * (2**100)


def test_vector_queries_sort_copy_equality_and_hash():
    value = vec(3, 1, 2, 1)

    assert value.nth(-1) == 1
    assert value.nth(99, "missing") == "missing"
    assert value.index(1) == 1
    assert value.index(1, 2) == 3
    assert value.count(1) == 2
    assert list(value.sort()) == [1, 1, 2, 3]
    assert list(value.sort(reverse=True)) == [3, 2, 1, 1]
    assert value.copy() is value
    assert value == vec(3, 1, 2, 1)
    assert hash(value) == hash(vec(3, 1, 2, 1))
    assert list(reversed(value)) == [1, 2, 1, 3]

    with pytest.raises(IndexError):
        value.nth(99)
    with pytest.raises(ValueError):
        value.index(99)
    with pytest.raises(IndexError):
        EMPTY_VECTOR.pop()


def test_transient_vector_batch_workflow_and_invalidation():
    original = vec(*range(70))
    transient = original.transient()

    transient.append(70)
    transient.extend([71, 72])
    transient.assoc_mut(0, 100)
    transient[-1] = 720
    transient.pop_mut()

    result = transient.persistent()
    assert original[0] == 0
    assert len(original) == 70
    assert result[0] == 100
    assert list(result[-2:]) == [70, 71]

    with pytest.raises(RuntimeError):
        transient.append(73)
    with pytest.raises(RuntimeError):
        transient[0]


def test_transient_vector_sort():
    transient = vec(3, 1, 4, 2).transient()
    transient.sort(reverse=True)
    assert list(transient.persistent()) == [4, 3, 2, 1]


def test_vector_pickle_round_trip():
    value = vec(*range(100))
    restored = pickle.loads(pickle.dumps(value))

    assert isinstance(restored, Vector)
    assert restored == value
