from collections.abc import Mapping, MutableMapping, MutableSequence, MutableSet
from collections.abc import Sequence, Set as AbstractSet
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
