from collections.abc import Mapping, MutableMapping, MutableSequence, MutableSet
from collections.abc import Sequence, Set as AbstractSet
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import os
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


def test_concurrent_first_and_cached_hash_publication():
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
    rounds = 20
    barrier = Barrier(worker_count)

    def compute_hashes(_worker_id):
        results = []
        for _ in range(rounds):
            barrier.wait(timeout=10)
            results.append(tuple(hash(value) for value in cached_values))
        return results

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(compute_hashes, range(worker_count)))

    assert results == [results[0]] * worker_count
    assert results[0] == [results[0][0]] * rounds
    assert tuple(hash(value) for value in cached_values) == results[0][0]


def test_concurrent_reads_updates_and_independent_transients():
    vector = pds.Vector(range(100))
    mapping = pds.Map((index, index * 2) for index in range(100))
    set_value = pds.Set(range(100))
    sorted_value = pds.SortedVector(range(100))
    double_vector = pds.DoubleVector(range(100))
    int_vector = pds.IntVector(range(100))
    equal_values = (
        pds.Vector(range(100)),
        pds.Map((index, index * 2) for index in range(100)),
        pds.Set(range(100)),
        pds.SortedVector(range(100)),
    )
    worker_count = 8
    barrier = Barrier(worker_count)

    def exercise(worker_id):
        barrier.wait(timeout=10)
        for _ in range(50):
            assert vector[50] == 50
            assert mapping[50] == 100
            assert 50 in set_value
            assert sorted_value[50] == 50
            assert double_vector[50] == 50.0
            assert int_vector[50] == 50
            assert list(vector)[-1] == 99
            assert dict(mapping.items())[50] == 100
            assert set(set_value) == set(range(100))
            assert list(sorted_value)[0] == 0
            assert list(double_vector)[-1] == 99.0
            assert list(int_vector)[-1] == 99
            assert (
                vector,
                mapping,
                set_value,
                sorted_value,
            ) == equal_values
            assert memoryview(double_vector)[50] == 50.0
            assert memoryview(int_vector)[50] == 50
            hash(vector)
            hash(mapping)
            hash(set_value)
            hash(sorted_value)
            hash(double_vector)
            hash(int_vector)

        updated_vector = vector.assoc(50, worker_id)
        updated_map = mapping.assoc(50, worker_id)
        updated_set = set_value.conj(worker_id + 100)
        updated_sorted = sorted_value.conj(worker_id + 100)
        updated_doubles = double_vector.conj(worker_id + 100)
        updated_ints = int_vector.conj(worker_id + 100)

        vector_builder = vector.transient()
        vector_builder.append(worker_id)
        map_builder = mapping.transient()
        map_builder["worker"] = worker_id
        set_builder = set_value.transient()
        set_builder.add(worker_id + 100)
        return (
            updated_vector[50],
            updated_map[50],
            worker_id + 100 in updated_set,
            updated_sorted[-1],
            updated_doubles[-1],
            updated_ints[-1],
            vector_builder.persistent()[-1],
            map_builder.persistent()["worker"],
            worker_id + 100 in set_builder.persistent(),
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(exercise, range(worker_count)))

    assert results == [
        (
            worker_id,
            worker_id,
            True,
            worker_id + 100,
            float(worker_id + 100),
            worker_id + 100,
            worker_id,
            worker_id,
            True,
        )
        for worker_id in range(worker_count)
    ]
    assert list(vector) == list(range(100))
    assert dict(mapping.items()) == {index: index * 2 for index in range(100)}
    assert set(set_value) == set(range(100))


def _clean_subprocess_env(*, force_no_gil=False):
    env = os.environ.copy()
    env.pop("PYTHON_GIL", None)
    env.pop("PYTHONWARNINGS", None)
    if force_no_gil and sysconfig.get_config_var("Py_GIL_DISABLED"):
        env["PYTHON_GIL"] = "0"
    return env


def test_free_threaded_import_keeps_the_gil_disabled():
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("requires a free-threaded CPython build")

    assert not sys._is_gil_enabled()
    script = textwrap.dedent(
        """
        import sys
        import sysconfig

        assert sysconfig.get_config_var("Py_GIL_DISABLED")
        assert not sys._is_gil_enabled(), "GIL enabled before extension import"

        import spork_pds
        assert not sys._is_gil_enabled(), "GIL enabled by spork_pds import"

        import spork.pds
        assert not sys._is_gil_enabled(), "GIL enabled by spork.pds import"
        """
    )
    for env in (
        _clean_subprocess_env(),
        _clean_subprocess_env(force_no_gil=True),
    ):
        result = subprocess.run(
            [sys.executable, "-W", "error::RuntimeWarning", "-c", script],
            capture_output=True,
            env=env,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_concurrent_fresh_import_publishes_one_singleton_set():
    script = textwrap.dedent(
        """
        import importlib
        import sys
        import sysconfig
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        worker_count = 16
        barrier = Barrier(worker_count)

        def load(_worker_id):
            barrier.wait(timeout=10)
            module = importlib.import_module("spork_pds")
            return tuple(
                id(getattr(module, name))
                for name in (
                    "EMPTY_VECTOR",
                    "EMPTY_DOUBLE_VECTOR",
                    "EMPTY_LONG_VECTOR",
                    "EMPTY_MAP",
                    "EMPTY_SET",
                    "EMPTY_SORTED_VECTOR",
                )
            )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(load, range(worker_count)))

        assert results == [results[0]] * worker_count
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            assert not sys._is_gil_enabled()
        """
    )
    result = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", script],
        capture_output=True,
        env=_clean_subprocess_env(force_no_gil=True),
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_module_rejects_or_survives_legacy_subinterpreter_teardown():
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
            try:
                run(interpreter)
            except Exception as exc:
                # CPython 3.12 rejects the unsupported import. Newer APIs can
                # explicitly create a legacy shared-GIL interpreter that
                # bypasses the compatibility check.
                assert "does not support loading in subinterpreters" in str(exc)
            finally:
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
