#!/usr/bin/env python3
"""Fuzz testing for Vector implementation.

This tests the Vector implementation against a reference Python list
using random operations to find edge cases and verify correctness.
"""

import random
from typing import Any

from spork_pds import EMPTY_VECTOR, Vector

from .fuzz import Fuzzer, random_value


class VectorFuzzer(Fuzzer):
    """Fuzz tester that maintains a Vector and reference list."""

    name = "Vector"

    def __init__(self):
        super().__init__()
        self.vector: Vector = EMPTY_VECTOR
        self.reference: list = []
        self.old_versions: list[tuple[Vector, list]] = []
        self.max_size = 0

    def reset(self):
        """Reset state for a new example."""
        self.vector = EMPTY_VECTOR
        self.reference = []
        self.old_versions.clear()

    def release_all(self):
        """Release all references to allow GC."""
        self.vector = EMPTY_VECTOR
        self.reference = []
        self.old_versions.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return additional stats to display."""
        return {"Max vector size": self.max_size}

    def save_version(self):
        """Save current version for persistence checking."""
        self.old_versions.append((self.vector, self.reference.copy()))
        # Keep only last 20 versions
        if len(self.old_versions) > 20:
            self.old_versions = self.old_versions[-10:]

    def check_invariants(self):
        """Verify vector matches reference."""
        # Length check
        assert len(self.vector) == len(self.reference), (
            f"Length mismatch: {len(self.vector)} vs {len(self.reference)}"
        )

        # Content check
        vec_list = list(self.vector)
        assert vec_list == self.reference, (
            f"Content mismatch:\n  Vector: {vec_list[:10]}...\n  "
            f"Reference: {self.reference[:10]}..."
        )

        # Spot check indexing
        if len(self.reference) > 0:
            for i in [0, len(self.reference) - 1, len(self.reference) // 2]:
                if i < len(self.reference):
                    assert self.vector[i] == self.reference[i]
                    assert (
                        self.vector[i - len(self.reference)]
                        == self.reference[i - len(self.reference)]
                    )

        # Check old versions unchanged (persistence)
        for old_vec, old_ref in self.old_versions[-5:]:
            assert list(old_vec) == old_ref, "Persistence violation!"

    def do_conj(self):
        """Append a value."""
        self.save_version()
        value = random_value()
        self.vector = self.vector.conj(value)
        self.reference.append(value)
        self.record_op("conj")

    def do_conj_multiple(self):
        """Append multiple values."""
        self.save_version()
        count = random.randint(1, 20)
        for _ in range(count):
            value = random_value()
            self.vector = self.vector.conj(value)
            self.reference.append(value)
        self.record_op("conj_multi")

    def do_pop(self):
        """Remove last element."""
        if len(self.reference) == 0:
            return
        self.save_version()
        self.vector = self.vector.pop()
        self.reference.pop()
        self.record_op("pop")

    def do_pop_multiple(self):
        """Remove multiple elements."""
        if len(self.reference) == 0:
            return
        self.save_version()
        count = min(random.randint(1, 10), len(self.reference))
        for _ in range(count):
            self.vector = self.vector.pop()
            self.reference.pop()
        self.record_op("pop_multi")

    def do_assoc(self):
        """Update an element."""
        if len(self.reference) == 0:
            return
        self.save_version()
        idx = random.randint(0, len(self.reference) - 1)
        value = random_value()
        self.vector = self.vector.assoc(idx, value)
        self.reference[idx] = value
        self.record_op("assoc")

    def do_assoc_end(self):
        """Append using assoc at end."""
        self.save_version()
        value = random_value()
        self.vector = self.vector.assoc(len(self.reference), value)
        self.reference.append(value)
        self.record_op("assoc_end")

    def do_transient_extend(self):
        """Extend using transient."""
        self.save_version()
        t = self.vector.transient()
        count = random.randint(1, 50)
        for _ in range(count):
            value = random_value()
            t.conj_mut(value)
            self.reference.append(value)
        self.vector = t.persistent()
        self.record_op("t_extend")

    def do_transient_assoc(self):
        """Multiple assocs using transient."""
        if len(self.reference) == 0:
            return
        self.save_version()
        t = self.vector.transient()
        count = min(random.randint(1, 10), len(self.reference))
        for _ in range(count):
            idx = random.randint(0, len(self.reference) - 1)
            value = random_value()
            t.assoc_mut(idx, value)
            self.reference[idx] = value
        self.vector = t.persistent()
        self.record_op("t_assoc")

    def do_concat(self):
        """Concatenate with another vector."""
        self.save_version()
        other_values = [random_value() for _ in range(random.randint(0, 30))]
        other_vec = EMPTY_VECTOR
        for v in other_values:
            other_vec = other_vec.conj(v)
        self.vector = self.vector + other_vec
        self.reference.extend(other_values)
        self.record_op("concat")

    def do_random_operation(self):
        """Do a random operation."""
        ops = [
            (self.do_conj, 15),
            (self.do_conj_multiple, 10),
            (self.do_pop, 10),
            (self.do_pop_multiple, 5),
            (self.do_assoc, 15),
            (self.do_assoc_end, 10),
            (self.do_transient_extend, 15),
            (self.do_transient_assoc, 10),
            (self.do_concat, 10),
        ]

        total_weight = sum(w for _, w in ops)
        r = random.randint(1, total_weight)
        cumulative = 0
        for op, weight in ops:
            cumulative += weight
            if r <= cumulative:
                op()
                break

        self.max_size = max(self.max_size, len(self.reference))
