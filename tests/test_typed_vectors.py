import pickle

import pytest

from spork.pds import (
    EMPTY_DOUBLE_VECTOR,
    EMPTY_LONG_VECTOR,
    DoubleVector,
    IntVector,
    vec_f64,
    vec_i64,
)


def test_double_vector_operations_and_buffer():
    value = vec_f64(1, 2.5, 3)
    appended = value.conj(4)
    default = object()

    assert isinstance(value, DoubleVector)
    assert list(value) == [1.0, 2.5, 3.0]
    assert value[-1] == 3.0
    assert value.nth(99, default) is default
    assert isinstance(value[::2], DoubleVector)
    assert list(value[::2]) == [1.0, 3.0]
    assert list(value[::-1]) == [3.0, 2.5, 1.0]
    assert list(appended) == [1.0, 2.5, 3.0, 4.0]
    assert list(value) == [1.0, 2.5, 3.0]
    assert hash(value) == hash(vec_f64(1, 2.5, 3))
    assert repr(value) == "vec_f64([1.0, 2.5, 3.0])"

    with pytest.raises(IndexError):
        value[3]
    with pytest.raises(IndexError):
        value.nth(-4)
    with pytest.raises(TypeError):
        value["0"]

    buffer = memoryview(value)
    assert buffer.readonly
    assert buffer.format == "d"
    assert buffer.ndim == 1
    assert buffer.shape == (3,)
    assert buffer.tolist() == [1.0, 2.5, 3.0]

    with pytest.raises(TypeError):
        vec_f64("not-a-number")


def test_int_vector_operations_boundaries_and_buffer():
    values = [-(2**63), -1, 0, 1, 2**63 - 1]
    value = vec_i64(*values)
    appended = value.conj(42)
    default = object()

    assert isinstance(value, IntVector)
    assert list(value) == values
    assert value[-1] == 2**63 - 1
    assert value.nth(99, default) is default
    assert isinstance(value[1:4], IntVector)
    assert list(value[1:4]) == [-1, 0, 1]
    assert list(value[::-1]) == list(reversed(values))
    assert list(appended) == values + [42]
    assert hash(value) == hash(vec_i64(*values))
    assert repr(value) == f"vec_i64([{', '.join(map(str, values))}])"

    with pytest.raises(IndexError):
        value[len(value)]
    with pytest.raises(IndexError):
        value.nth(-len(value) - 1)
    with pytest.raises(TypeError):
        value["0"]

    buffer = memoryview(value)
    assert buffer.readonly
    assert buffer.format == "q"
    assert buffer.ndim == 1
    assert buffer.shape == (len(values),)
    assert buffer.tolist() == values

    with pytest.raises(TypeError):
        vec_i64(1.5)
    with pytest.raises(TypeError):
        vec_i64(2**63)


def test_typed_vector_class_constructors_and_argument_errors():
    assert list(DoubleVector(range(3))) == [0.0, 1.0, 2.0]
    assert list(DoubleVector(1, 2.5)) == [1.0, 2.5]
    assert list(IntVector(range(3))) == [0, 1, 2]
    assert list(IntVector(1, 2)) == [1, 2]

    with pytest.raises(TypeError):
        DoubleVector(values=[1, 2])
    with pytest.raises(TypeError):
        IntVector(values=[1, 2])
    with pytest.raises(TypeError):
        DoubleVector("not-a-number")
    with pytest.raises(TypeError):
        IntVector(1.5)


def test_large_typed_vectors_cross_trie_boundaries():
    doubles = vec_f64(*range(1100))
    integers = vec_i64(*range(1100))

    assert len(doubles) == len(integers) == 1100
    for index in (0, 31, 32, 33, 1023, 1024, 1055, 1056, 1099):
        assert doubles[index] == float(index)
        assert integers[index] == index
    assert sum(doubles) == sum(range(1100))
    assert sum(integers) == sum(range(1100))

    assert memoryview(doubles).tolist() == [float(value) for value in range(1100)]
    assert memoryview(integers).tolist() == list(range(1100))


def test_typed_vector_transients_and_invalidation():
    original_doubles = vec_f64(*range(40))
    double_transient = original_doubles.transient()
    assert double_transient.conj_mut(40) is double_transient
    for value in range(41, 1100):
        double_transient.conj_mut(value)
    doubles = double_transient.persistent()
    assert list(doubles) == [float(value) for value in range(1100)]
    assert list(original_doubles) == [float(value) for value in range(40)]

    original_integers = vec_i64(*range(40))
    int_transient = original_integers.transient()
    assert int_transient.conj_mut(40) is int_transient
    for value in range(41, 1100):
        int_transient.conj_mut(value)
    integers = int_transient.persistent()
    assert list(integers) == list(range(1100))
    assert list(original_integers) == list(range(40))

    for operation in (
        lambda: double_transient.conj_mut(1100),
        lambda: double_transient.persistent(),
        lambda: int_transient.conj_mut(1100),
        lambda: int_transient.persistent(),
    ):
        with pytest.raises(RuntimeError):
            operation()


def test_empty_typed_vector_buffers_and_read_only_enforcement():
    for value, expected_format in (
        (EMPTY_DOUBLE_VECTOR, "d"),
        (EMPTY_LONG_VECTOR, "q"),
    ):
        view = memoryview(value)
        assert view.format == expected_format
        assert view.readonly
        assert view.shape == (0,)
        assert view.nbytes == 0
        assert view.tolist() == []

    value = vec_i64(1, 2, 3)
    first = memoryview(value)
    second = memoryview(value)
    assert first.obj is second.obj is value
    with pytest.raises(TypeError):
        first[0] = 10


def test_typed_vector_pickle_round_trip():
    doubles = vec_f64(1, 2.5, 3)
    integers = vec_i64(1, 2, 3)

    restored_doubles = pickle.loads(pickle.dumps(doubles))
    restored_integers = pickle.loads(pickle.dumps(integers))

    assert isinstance(restored_doubles, DoubleVector)
    assert isinstance(restored_integers, IntVector)
    assert list(restored_doubles) == list(doubles)
    assert list(restored_integers) == list(integers)


def test_numpy_views_are_read_only():
    np = pytest.importorskip("numpy")
    value = vec_f64(1, 2, 3)
    array = np.asarray(value)

    assert array.tolist() == [1.0, 2.0, 3.0]
    assert not array.flags.writeable
