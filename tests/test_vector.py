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
    updated_from_end = original.assoc(-1, "last")
    assoc_at_end = original.assoc(len(original), 80)
    popped = appended.pop()

    assert list(original) == list(range(80))
    assert appended[-1] == 80
    assert updated[40] == "changed"
    assert updated_from_end[-1] == "last"
    assert original[40] == 40
    assert list(assoc_at_end) == list(appended)
    assert list(popped) == list(original)
    assert list(vec(1, 2) + vec(3, 4)) == [1, 2, 3, 4]

    with pytest.raises(IndexError):
        original.assoc(-len(original) - 1, None)
    with pytest.raises(IndexError):
        original.assoc(len(original) + 1, None)


def test_vector_trie_boundaries_and_repeated_pops():
    values = list(range(1100))
    value = Vector(values)
    snapshot = value

    for index in (0, 31, 32, 33, 1023, 1024, 1055, 1056, 1099):
        assert value[index] == values[index]

    for expected_length in range(len(values) - 1, -1, -1):
        value = value.pop()
        assert len(value) == expected_length
        if value:
            assert value[-1] == expected_length - 1

    assert value is EMPTY_VECTOR
    assert list(snapshot) == values


def test_vector_slices_match_python_lists():
    values = list(range(75))
    value = Vector(values)
    slices = [
        slice(None),
        slice(None, None, -1),
        slice(5, 60, 7),
        slice(-30, -2, 3),
        slice(60, 5, -4),
        slice(1000, 2000),
    ]

    for item in slices:
        result = value[item]
        assert isinstance(result, Vector)
        assert list(result) == values[item]

    with pytest.raises(TypeError):
        value[1.5]
    with pytest.raises(IndexError):
        value[len(value)]
    with pytest.raises(IndexError):
        value[-len(value) - 1]


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
    default = object()

    assert value.nth(-1) == 1
    assert value.nth(99, default) is default
    assert value.index(1) == 1
    assert value.index(1, 2) == 3
    assert value.index(1, -2) == 3
    assert value.count(1) == 2
    assert list(value.sort()) == [1, 1, 2, 3]
    assert list(value.sort(reverse=True)) == [3, 2, 1, 1]
    assert list(vec("aaa", "b", "cc").sort(key=len)) == ["b", "cc", "aaa"]
    assert value.copy() is value
    assert value == vec(3, 1, 2, 1)
    assert value != vec(3, 1, 2)
    assert value != [3, 1, 2, 1]
    assert hash(value) == hash(vec(3, 1, 2, 1))
    assert list(reversed(value)) == [1, 2, 1, 3]

    with pytest.raises(IndexError):
        value.nth(99)
    with pytest.raises(ValueError):
        value.index(99)
    with pytest.raises(ValueError):
        value.index(1, 2, 3)
    with pytest.raises(TypeError):
        hash(Vector([[]]))
    with pytest.raises(IndexError):
        EMPTY_VECTOR.pop()


def test_vector_repr_and_cons_conversion():
    value = vec(1, "two", None)

    assert repr(value) == "[1 'two' None]"
    assert list(value.to_seq()) == [1, "two", None]
    assert EMPTY_VECTOR.to_seq() is None


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


def test_transient_vector_mutable_sequence_protocol():
    transient = vec(*range(40)).transient()

    assert transient.conj_mut(40) is transient
    assert transient.assoc_mut(-1, 400) is transient
    assert transient.assoc_mut(len(transient), 41) is transient
    assert transient.extend(iter([42, 43])) is transient
    assert transient[-1] == 43
    assert 400 in transient

    del transient[-1]
    with pytest.raises(NotImplementedError):
        del transient[0]
    with pytest.raises(IndexError):
        transient.assoc_mut(len(transient) + 1, 0)
    with pytest.raises(IndexError):
        transient[len(transient)] = 0
    with pytest.raises(TypeError):
        transient[1:2]

    assert list(transient.persistent()) == list(range(40)) + [400, 41, 42]


def test_transient_vector_sort():
    transient = vec("aaa", "b", "cc").transient()
    assert transient.sort(key=len, reverse=True) is None
    assert list(transient.persistent()) == ["aaa", "cc", "b"]


def test_transient_vector_all_access_is_invalid_after_persistent():
    transient = vec(1, 2, 3).transient()
    iterator = iter(transient)
    transient.persistent()

    operations = [
        lambda: len(transient),
        lambda: transient[0],
        lambda: 1 in transient,
        lambda: iter(transient),
        lambda: next(iterator),
        lambda: transient.append(4),
        lambda: transient.extend([4]),
        lambda: transient.assoc_mut(0, 4),
        lambda: transient.pop_mut(),
        lambda: transient.sort(),
        lambda: transient.persistent(),
    ]
    for operation in operations:
        with pytest.raises(RuntimeError):
            operation()


def test_vector_pickle_round_trip():
    value = vec(*range(100))
    restored = pickle.loads(pickle.dumps(value))

    assert isinstance(restored, Vector)
    assert restored == value
