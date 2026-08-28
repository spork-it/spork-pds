import bisect
import random

from spork_pds import EMPTY_MAP, EMPTY_SET, EMPTY_VECTOR, sorted_vec


class CollisionKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return 7

    def __eq__(self, other):
        return isinstance(other, CollisionKey) and self.value == other.value

    def __repr__(self):
        return f"CollisionKey({self.value})"


def test_vector_operations_match_a_list_model():
    rng = random.Random(0x5EED)
    value = EMPTY_VECTOR
    reference = []
    snapshots = []

    for _ in range(800):
        snapshots.append((value, reference.copy()))
        snapshots = snapshots[-12:]
        choice = rng.random()

        if not reference or (choice < 0.32 and len(reference) < 300):
            item = rng.randrange(-1000, 1000)
            value = value.conj(item)
            reference.append(item)
        elif choice < 0.52:
            value = value.pop()
            reference.pop()
        elif choice < 0.72:
            index = rng.randrange(-len(reference), len(reference))
            item = rng.randrange(-1000, 1000)
            value = value.assoc(index, item)
            reference[index] = item
        elif choice < 0.88:
            transient = value.transient()
            for _ in range(rng.randrange(1, 8)):
                if reference and rng.random() < 0.3:
                    transient.pop_mut()
                    reference.pop()
                else:
                    item = rng.randrange(-1000, 1000)
                    transient.conj_mut(item)
                    reference.append(item)
            value = transient.persistent()
        else:
            extension = [rng.randrange(-1000, 1000) for _ in range(rng.randrange(8))]
            value = value + iter(extension)
            reference.extend(extension)

        assert list(value) == reference
        assert len(value) == len(reference)
        for old_value, old_reference in snapshots[-3:]:
            assert list(old_value) == old_reference


def test_map_operations_match_a_dict_model():
    rng = random.Random(0xC0111DE)
    keys = list(range(80)) + [CollisionKey(index) for index in range(20)]
    value = EMPTY_MAP
    reference = {}
    snapshots = []

    for _ in range(1000):
        snapshots.append((value, reference.copy()))
        snapshots = snapshots[-12:]
        choice = rng.random()

        if choice < 0.6:
            key = rng.choice(keys)
            item = rng.randrange(-1000, 1000)
            value = value.assoc(key, item)
            reference[key] = item
        elif choice < 0.8:
            key = rng.choice(keys)
            value = value.dissoc(key)
            reference.pop(key, None)
        else:
            transient = value.transient()
            for _ in range(rng.randrange(1, 8)):
                key = rng.choice(keys)
                if rng.random() < 0.65:
                    item = rng.randrange(-1000, 1000)
                    transient.assoc_mut(key, item)
                    reference[key] = item
                else:
                    transient.dissoc_mut(key)
                    reference.pop(key, None)
            value = transient.persistent()

        assert dict(value.items()) == reference
        assert len(value) == len(reference)
        for old_value, old_reference in snapshots[-3:]:
            assert dict(old_value.items()) == old_reference


def test_set_operations_match_a_builtin_set_model():
    rng = random.Random(0x5E7)
    items = list(range(80)) + [CollisionKey(index) for index in range(20)]
    value = EMPTY_SET
    reference = set()
    snapshots = []

    for _ in range(1000):
        snapshots.append((value, reference.copy()))
        snapshots = snapshots[-12:]
        choice = rng.random()

        if choice < 0.6:
            item = rng.choice(items)
            value = value.conj(item)
            reference.add(item)
        elif choice < 0.8:
            item = rng.choice(items)
            value = value.disj(item)
            reference.discard(item)
        else:
            transient = value.transient()
            for _ in range(rng.randrange(1, 8)):
                item = rng.choice(items)
                if rng.random() < 0.6:
                    transient.conj_mut(item)
                    reference.add(item)
                else:
                    transient.disj_mut(item)
                    reference.discard(item)
            value = transient.persistent()

        assert set(value) == reference
        assert len(value) == len(reference)
        for old_value, old_reference in snapshots[-3:]:
            assert set(old_value) == old_reference


def test_sorted_vector_operations_match_a_sorted_list_model():
    rng = random.Random(0x50A7ED)
    value = sorted_vec()
    reference = []
    snapshots = []

    for _ in range(800):
        snapshots.append((value, reference.copy()))
        snapshots = snapshots[-12:]
        item = rng.randrange(40)
        choice = rng.random()

        if not reference or choice < 0.55:
            value = value.conj(item)
            bisect.insort(reference, item)
        elif choice < 0.8:
            value = value.disj(item)
            if item in reference:
                reference.remove(item)
        else:
            transient = value.transient()
            for _ in range(rng.randrange(1, 8)):
                item = rng.randrange(40)
                if rng.random() < 0.55:
                    transient.conj_mut(item)
                    bisect.insort(reference, item)
                else:
                    transient.disj_mut(item)
                    if item in reference:
                        reference.remove(item)
            value = transient.persistent()

        assert list(value) == reference
        assert len(value) == len(reference)
        probe = rng.randrange(40)
        assert value.rank(probe) == bisect.bisect_left(reference, probe)
        expected_index = reference.index(probe) if probe in reference else -1
        assert value.index_of(probe) == expected_index
        for old_value, old_reference in snapshots[-3:]:
            assert list(old_value) == old_reference
