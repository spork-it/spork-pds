from collections.abc import Mapping, MutableMapping, MutableSequence, MutableSet
from collections.abc import Sequence, Set as AbstractSet

import spork_pds as pds


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
