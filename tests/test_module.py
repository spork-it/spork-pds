from collections.abc import Mapping, MutableMapping, MutableSequence, MutableSet
from collections.abc import Sequence, Set as AbstractSet
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import subprocess
import sys
import sysconfig
import textwrap
from typing import get_args, get_origin

import pytest

import spork.pds as pds
import spork_pds as legacy_pds


def test_legacy_module_reexports_the_same_objects():
    for name in pds.__all__:
        assert getattr(legacy_pds, name) is getattr(pds, name)


def test_public_exports_and_module_names():
    expected = {
        "Cons",
        "Vector",
        "TransientVector",
        "DoubleVector",
        "TransientDoubleVector",
        "IntVector",
        "TransientIntVector",
        "Map",
        "TransientMap",
        "Set",
        "TransientSet",
        "SortedVector",
        "TransientSortedVector",
        "EMPTY_VECTOR",
        "EMPTY_DOUBLE_VECTOR",
        "EMPTY_LONG_VECTOR",
        "EMPTY_MAP",
        "EMPTY_SET",
        "EMPTY_SORTED_VECTOR",
        "cons",
        "vec",
        "vec_f64",
        "vec_i64",
        "hash_map",
        "hash_set",
        "sorted_vec",
    }

    assert expected <= set(dir(pds))
    for type_name in expected & {
        "Cons",
        "Vector",
        "TransientVector",
        "DoubleVector",
        "TransientDoubleVector",
        "IntVector",
        "TransientIntVector",
        "Map",
        "TransientMap",
        "Set",
        "TransientSet",
        "SortedVector",
        "TransientSortedVector",
    }:
        assert getattr(pds, type_name).__module__ == "spork_pds"


def test_empty_factories_return_shared_values():
    assert pds.vec() is pds.EMPTY_VECTOR
    assert pds.vec_f64() is pds.EMPTY_DOUBLE_VECTOR
    assert pds.vec_i64() is pds.EMPTY_LONG_VECTOR
    assert pds.hash_map() is pds.EMPTY_MAP
    assert pds.hash_set() is pds.EMPTY_SET
    assert pds.sorted_vec() is not None
    assert len(pds.sorted_vec()) == len(pds.EMPTY_SORTED_VECTOR) == 0


def test_collections_abc_registration():
    assert isinstance(pds.vec(), Sequence)
    assert isinstance(pds.vec_f64(), Sequence)
    assert isinstance(pds.vec_i64(), Sequence)
    assert isinstance(pds.sorted_vec(), Sequence)
    assert isinstance(pds.cons(1), Sequence)
    assert isinstance(pds.hash_map(), Mapping)
    assert isinstance(pds.hash_set(), AbstractSet)

    assert isinstance(pds.vec().transient(), MutableSequence)
    assert isinstance(pds.hash_map().transient(), MutableMapping)
    assert isinstance(pds.hash_set().transient(), MutableSet)


def test_persistent_types_support_generic_aliases():
    aliases = [
        (pds.Cons[int], pds.Cons, (int,)),
        (pds.Vector[str], pds.Vector, (str,)),
        (pds.DoubleVector[float], pds.DoubleVector, (float,)),
        (pds.IntVector[int], pds.IntVector, (int,)),
        (pds.Map[str, int], pds.Map, (str, int)),
        (pds.Set[str], pds.Set, (str,)),
        (pds.SortedVector[int], pds.SortedVector, (int,)),
    ]

    for alias, origin, args in aliases:
        assert get_origin(alias) is origin
        assert get_args(alias) == args


def test_concurrent_first_hash_publication():
    cons_value = None
    for value in reversed(range(512)):
        cons_value = pds.cons(value, cons_value)

    cached_values = (
        cons_value,
        pds.Vector(range(2048)),
        pds.Map((value, value * 2) for value in range(512)),
        pds.Set(range(512)),
        pds.DoubleVector(range(2048)),
        pds.IntVector(range(2048)),
    )
    worker_count = 8
    barriers = [Barrier(worker_count) for _ in cached_values]

    def compute_hashes(_worker_id):
        results = []
        for barrier, value in zip(barriers, cached_values, strict=True):
            barrier.wait(timeout=10)
            results.append(hash(value))
        return tuple(results)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(compute_hashes, range(worker_count)))

    assert results == [results[0]] * worker_count
    assert tuple(hash(value) for value in cached_values) == results[0]


def test_concurrent_reads_and_independent_transients():
    vector = pds.Vector(range(100))
    mapping = pds.Map((index, index * 2) for index in range(100))
    set_value = pds.Set(range(100))
    sorted_value = pds.SortedVector(range(100))
    double_vector = pds.DoubleVector(range(100))
    int_vector = pds.IntVector(range(100))

    def exercise(worker_id):
        for _ in range(100):
            assert vector[50] == 50
            assert mapping[50] == 100
            assert 50 in set_value
            assert sorted_value[50] == 50
            assert double_vector[50] == 50.0
            assert int_vector[50] == 50
            assert memoryview(double_vector)[50] == 50.0
            assert memoryview(int_vector)[50] == 50
            hash(vector)
            hash(mapping)
            hash(set_value)
            hash(sorted_value)
            hash(double_vector)
            hash(int_vector)

        vector_builder = vector.transient()
        vector_builder.append(worker_id)
        map_builder = mapping.transient()
        map_builder["worker"] = worker_id
        set_builder = set_value.transient()
        set_builder.add(worker_id + 100)
        return (
            vector_builder.persistent()[-1],
            map_builder.persistent()["worker"],
            worker_id + 100 in set_builder.persistent(),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(exercise, range(8)))

    assert results == [(worker_id, worker_id, True) for worker_id in range(8)]


def test_free_threaded_build_uses_the_compatibility_gil():
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("requires a free-threaded CPython build")

    assert sys._is_gil_enabled()


def test_module_survives_legacy_subinterpreter_teardown():
    script = textwrap.dedent(
        """
        import spork_pds

        try:
            import _interpreters as interpreters
        except ImportError:
            try:
                import _xxsubinterpreters as interpreters
            except ImportError:
                raise SystemExit(0)

            def create():
                return interpreters.create()

            def run(interpreter):
                interpreters.run_string(
                    interpreter,
                    "import spork_pds; assert len(spork_pds.EMPTY_VECTOR) == 0",
                )
        else:
            def create():
                try:
                    return interpreters.create("legacy")
                except TypeError:
                    return interpreters.create()

            def run(interpreter):
                interpreters.exec(
                    interpreter,
                    "import spork_pds; assert len(spork_pds.EMPTY_VECTOR) == 0",
                )

        for _ in range(5):
            interpreter = create()
            run(interpreter)
            interpreters.destroy(interpreter)
            assert len(spork_pds.EMPTY_VECTOR) == 0
            assert len(spork_pds.EMPTY_MAP) == 0
            assert len(spork_pds.EMPTY_SET) == 0
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_transient_types_cannot_be_constructed_directly():
    transient_types = [
        pds.TransientVector,
        pds.TransientDoubleVector,
        pds.TransientIntVector,
        pds.TransientMap,
        pds.TransientSet,
        pds.TransientSortedVector,
    ]

    for transient_type in transient_types:
        with pytest.raises(TypeError):
            transient_type()
