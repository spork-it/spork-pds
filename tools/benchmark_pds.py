"""Benchmark spork-pds against Python's built-in mutable collections.

Usage:
    python tools/benchmark_pds.py --size 100000 --iter 20
"""

import argparse
import array
import gc
import os
import random
import sys
import time
from collections.abc import Callable

# Try importing standard libraries for comparison
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Import the spork-pds C extension
try:
    from spork_pds import (
        EMPTY_DOUBLE_VECTOR,
        EMPTY_LONG_VECTOR,
        EMPTY_MAP,
        EMPTY_SET,
        EMPTY_VECTOR,
        DoubleVector,
        IntVector,
        Map,
        Set,
        TransientVector,
        Vector,
        hash_map,
        vec,
        vec_f64,
        vec_i64,
    )

except ImportError:
    print("❌ Error: Could not import 'spork_pds'.")
    print(
        "   Make sure you have built the C extension (python setup.py build_ext --inplace)"
    )
    sys.exit(1)

# --- Utilities ---


_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


class Colors:
    HEADER = "\033[95m" if _USE_COLOR else ""
    BLUE = "\033[94m" if _USE_COLOR else ""
    GREEN = "\033[92m" if _USE_COLOR else ""  # Fastest / winner
    YELLOW = "\033[93m" if _USE_COLOR else ""  # Comparable (0.9x - 1.5x)
    ORANGE = "\033[38;5;208m" if _USE_COLOR else ""  # Moderately slower
    RED = "\033[91m" if _USE_COLOR else ""  # Slow (3x - 10x)
    DARK_RED = "\033[38;5;160m" if _USE_COLOR else ""  # Very slow (10x+)
    GRAY = "\033[90m" if _USE_COLOR else ""
    ENDC = "\033[0m" if _USE_COLOR else ""
    BOLD = "\033[1m" if _USE_COLOR else ""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.4f} s"


def run_benchmark(label: str, func: Callable, iterations: int) -> float:
    # Warmup
    func()

    # Collect garbage and disable GC during timing for fairness
    gc.collect()
    gc.disable()

    try:
        # Timing
        start = time.perf_counter()
        for _ in range(iterations):
            func()
        end = time.perf_counter()
    finally:
        gc.enable()

    avg_time = (end - start) / iterations
    return avg_time


def format_ratio(baseline_time: float, challenger_time: float) -> tuple[str, str]:
    """Returns (color, verdict_string) for a timing comparison."""
    ratio = challenger_time / baseline_time

    if ratio <= 1.1:
        # Essentially the same or faster
        color = Colors.GREEN
        if ratio < 0.95:
            verdict = f"{1 / ratio:.2f}x faster"
        else:
            verdict = "~same"
    elif ratio <= 1.5:
        # Slightly slower
        color = Colors.YELLOW
        verdict = f"{ratio:.2f}x slower"
    elif ratio <= 3.0:
        # Moderately slower
        color = Colors.ORANGE
        verdict = f"{ratio:.2f}x slower"
    elif ratio <= 10.0:
        # Significantly slower
        color = Colors.RED
        verdict = f"{ratio:.1f}x slower"
    else:
        # Very slow
        color = Colors.DARK_RED
        verdict = f"{ratio:.0f}x slower"

    return color, verdict


def print_comparison(
    baseline_name: str,
    baseline_time: float,
    challenger_name: str,
    challenger_time: float,
):
    """Legacy pairwise comparison (kept for compatibility)."""
    color, verdict = format_ratio(baseline_time, challenger_time)

    print(f"  {baseline_name:<25} : {format_time(baseline_time)}")
    print(
        f"  {color}{challenger_name:<25} : {format_time(challenger_time)} ({verdict}){Colors.ENDC}"
    )


def print_single(name: str, time_val: float):
    print(f"  {name:<25} : {format_time(time_val)}")


def print_group(title: str, results: list[tuple[str, float]]):
    """
    Print a group of benchmark results sorted fastest to slowest.

    Args:
        title: Section title
        results: List of (name, time) tuples
    """
    print(f"{Colors.BOLD}--- {title} ---{Colors.ENDC}")

    if not results:
        return

    # Sort by time (fastest first)
    sorted_results = sorted(results, key=lambda x: x[1])
    baseline_name, baseline_time = sorted_results[0]

    # Find max name length for alignment
    max_name_len = max(len(name) for name, _ in results)
    col_width = max(max_name_len + 2, 28)

    for i, (name, time_val) in enumerate(sorted_results):
        time_str = format_time(time_val)

        if i == 0:
            # Fastest row - marked as baseline
            print(
                f"  {Colors.GREEN}{name:<{col_width}} {time_str:>12}  (fastest){Colors.ENDC}"
            )
        else:
            # Compare to fastest
            color, verdict = format_ratio(baseline_time, time_val)
            print(
                f"  {color}{name:<{col_width}} {time_str:>12}  ({verdict}){Colors.ENDC}"
            )

    print()


# --- Benchmark Implementations ---


class Benchmarks:
    def __init__(self, size: int, iterations: int):
        self.N = size
        self.ITERS = iterations
        self.data_int = list(range(self.N))
        self.data_float = [float(x) for x in range(self.N)]
        self.data_str = [f"key_{x}" for x in range(self.N)]

        # Pre-built structures for read tests
        print(f"generating pre-built structures (N={self.N})...", end="", flush=True)
        self.py_list = list(self.data_int)
        self.py_dict = dict(zip(self.data_str, self.data_int))
        self.py_set = set(self.data_int)

        # Build spork-pds structures using transients for speed
        t_vec = EMPTY_VECTOR.transient()
        for i in self.data_int:
            t_vec.conj_mut(i)
        self.spork_vec = t_vec.persistent()

        t_map = EMPTY_MAP.transient()
        for k, v in zip(self.data_str, self.data_int):
            t_map.assoc_mut(k, v)
        self.spork_map = t_map.persistent()

        t_set = EMPTY_SET.transient()
        for i in self.data_int:
            t_set.conj_mut(i)
        self.spork_set = t_set.persistent()

        t_dvec = EMPTY_DOUBLE_VECTOR.transient()
        for f in self.data_float:
            t_dvec.conj_mut(f)
        self.spork_dvec = t_dvec.persistent()

        t_ivec = EMPTY_LONG_VECTOR.transient()
        for i in self.data_int:
            t_ivec.conj_mut(i)
        self.spork_ivec = t_ivec.persistent()

        print(" done.\n")

    # --- Vector Construction ---

    def bench_vector_construction(self):
        def py_list_append():
            l = []
            for i in range(self.N):
                l.append(i)
            return l

        def py_list_comprehension():
            return [i for i in range(self.N)]

        def py_list_from_range():
            return list(range(self.N))

        def spork_transient():
            t = EMPTY_VECTOR.transient()
            for i in range(self.N):
                t.conj_mut(i)
            return t.persistent()

        def spork_persistent():
            v = EMPTY_VECTOR
            for i in range(self.N):
                v = v.conj(i)
            return v

        def spork_vec_factory():
            return vec(*range(self.N))

        py_append_time = run_benchmark("list.append", py_list_append, self.ITERS)
        py_comp_time = run_benchmark(
            "list comprehension", py_list_comprehension, self.ITERS
        )
        py_range_time = run_benchmark("list(range)", py_list_from_range, self.ITERS)
        spork_t_time = run_benchmark("TransientVector", spork_transient, self.ITERS)
        spork_p_time = run_benchmark(
            "Vector.conj", spork_persistent, max(1, self.ITERS // 10)
        )
        spork_factory_time = run_benchmark("vec(*range)", spork_vec_factory, self.ITERS)

        print_group(
            f"Vector Construction (N={self.N})",
            [
                ("Python list(range(N))", py_range_time),
                ("Python [x for x in range(N)]", py_comp_time),
                ("Python list.append() loop", py_append_time),
                ("spork-pds vec(*range(N))", spork_factory_time),
                ("spork-pds TransientVector", spork_t_time),
                ("spork-pds Vector.conj() chain", spork_p_time),
            ],
        )

    # def bench_vector_persistent_construction(self):
    #     print(
    #         f"{Colors.BOLD}--- Vector Persistent Construction (Functional Append) ---{Colors.ENDC}"
    #     )

    #     def py_list_copy_append():
    #         l = []
    #         for i in range(self.N):
    #             l = l + [i]  # Creates new list each time
    #         return l

    #     def spork_persistent():
    #         v = EMPTY_VECTOR
    #         for i in range(self.N):
    #             v = v.conj(i)
    #         return v

    #     py_time = run_benchmark("list + [x]", py_list_copy_append, self.ITERS)
    #     spork_time = run_benchmark("Vector.conj", spork_persistent, self.ITERS)

    #     print_comparison("Python list copy", py_time, "spork-pds persistent", spork_time)

    # --- Specialized Vector Construction ---

    def bench_typed_vector_construction(self):
        def py_list_float():
            l = []
            for f in self.data_float:
                l.append(f)
            return l

        def py_array_float():
            a = array.array("d")
            a.extend(self.data_float)
            return a

        def spork_transient_f64():
            t = EMPTY_DOUBLE_VECTOR.transient()
            for f in self.data_float:
                t.conj_mut(f)
            return t.persistent()

        def spork_vec_f64_factory():
            return vec_f64(*self.data_float)

        def spork_vec_box_factory():
            return vec(*self.data_float)

        def py_list_int():
            l = []
            for i in self.data_int:
                l.append(i)
            return l

        def py_array_int():
            a = array.array("q")
            a.extend(self.data_int)
            return a

        def spork_transient_i64():
            t = EMPTY_LONG_VECTOR.transient()
            for i in self.data_int:
                t.conj_mut(i)
            return t.persistent()

        def spork_vec_i64_factory():
            return vec_i64(*self.data_int)

        def spork_vec_box_factory():
            return vec(*self.data_int)

        py_float_time = run_benchmark("list[float]", py_list_float, self.ITERS)
        arr_float_time = run_benchmark("array('d')", py_array_float, self.ITERS)
        spork_f64_time = run_benchmark("DoubleVector", spork_transient_f64, self.ITERS)
        spork_f64_factory_time = run_benchmark(
            "vec_f64(*data)", spork_vec_f64_factory, self.ITERS
        )
        # also test regular spork vec(*data) for floats
        spork_f64_vec_box_time = run_benchmark(
            "vec(*data)", spork_vec_box_factory, self.ITERS
        )

        py_int_time = run_benchmark("list[int]", py_list_int, self.ITERS)
        arr_int_time = run_benchmark("array('q')", py_array_int, self.ITERS)
        spork_i64_time = run_benchmark("IntVector", spork_transient_i64, self.ITERS)
        spork_i64_factory_time = run_benchmark(
            "vec_i64(*data)", spork_vec_i64_factory, self.ITERS
        )
        spork_i64_vec_box_time = run_benchmark(
            "vec(*data)", spork_vec_box_factory, self.ITERS
        )

        print_group(
            "Float64 Vector Construction",
            [
                ("Python list[float]", py_float_time),
                ("Python array('d').extend()", arr_float_time),
                ("spork-pds TransientDoubleVector", spork_f64_time),
                ("spork-pds vec_f64(*data)", spork_f64_factory_time),
                ("spork-pds vec(*data) boxed", spork_f64_vec_box_time),
            ],
        )

        print_group(
            "Int64 Vector Construction",
            [
                ("Python list[int]", py_int_time),
                ("Python array('q').extend()", arr_int_time),
                ("spork-pds TransientIntVector", spork_i64_time),
                ("spork-pds vec_i64(*data)", spork_i64_factory_time),
                ("spork-pds vec(*data) boxed", spork_i64_vec_box_time),
            ],
        )

    # --- Vector Access ---

    def bench_vector_access(self):
        num_accesses = min(10000, self.N)
        indices = [random.randint(0, self.N - 1) for _ in range(num_accesses)]

        def read_py_list():
            s = 0
            for idx in indices:
                s += self.py_list[idx]
            return s

        def read_spork_vec():
            s = 0
            for idx in indices:
                s += self.spork_vec.nth(idx)
            return s

        def read_spork_vec_getitem():
            s = 0
            for idx in indices:
                s += self.spork_vec[idx]
            return s

        def iter_py_list():
            s = 0
            for x in self.py_list:
                s += x
            return s

        def iter_spork_vec():
            s = 0
            for x in self.spork_vec:
                s += x
            return s

        def iter_spork_dvec():
            s = 0.0
            for x in self.spork_dvec:
                s += x
            return s

        def iter_spork_ivec():
            s = 0
            for x in self.spork_ivec:
                s += x
            return s

        py_rand_time = run_benchmark("list[i]", read_py_list, self.ITERS)
        spork_rand_time = run_benchmark("Vector.nth", read_spork_vec, self.ITERS)
        spork_getitem_time = run_benchmark(
            "Vector[i]", read_spork_vec_getitem, self.ITERS
        )

        py_seq_time = run_benchmark("list iteration", iter_py_list, self.ITERS)
        spork_seq_time = run_benchmark("Vector iteration", iter_spork_vec, self.ITERS)
        spork_dvec_time = run_benchmark(
            "DoubleVector iter", iter_spork_dvec, self.ITERS
        )
        spork_ivec_time = run_benchmark("IntVector iter", iter_spork_ivec, self.ITERS)

        print_group(
            f"Random Access ({num_accesses} reads)",
            [
                ("Python list[i]", py_rand_time),
                ("spork-pds Vector.nth(i)", spork_rand_time),
                ("spork-pds Vector[i]", spork_getitem_time),
            ],
        )

        print_group(
            "Sequential Iteration",
            [
                ("Python list", py_seq_time),
                ("spork-pds Vector", spork_seq_time),
                ("spork-pds DoubleVector", spork_dvec_time),
                ("spork-pds IntVector", spork_ivec_time),
            ],
        )

    # --- Vector Pop ---

    def bench_vector_pop(self):
        pop_count = min(1000, self.N // 10)

        def py_list_pop():
            l = self.py_list.copy()
            for _ in range(pop_count):
                l.pop()
            return l

        def spork_transient_pop():
            t = self.spork_vec.transient()
            for _ in range(pop_count):
                t.pop_mut()
            return t.persistent()

        def spork_persistent_pop():
            v = self.spork_vec
            for _ in range(pop_count):
                v = v.pop()
            return v

        py_time = run_benchmark("list.pop()", py_list_pop, self.ITERS)
        spork_t_time = run_benchmark(
            "Transient.pop_mut", spork_transient_pop, self.ITERS
        )
        spork_p_time = run_benchmark("Vector.pop", spork_persistent_pop, self.ITERS)

        print_group(
            f"Vector Pop ({pop_count} pops)",
            [
                ("Python list.pop()", py_time),
                ("spork-pds Transient.pop_mut()", spork_t_time),
                ("spork-pds Vector.pop()", spork_p_time),
            ],
        )

    # --- Map Operations ---

    def bench_map_construction(self):
        def py_dict_build():
            d = {}
            for k, v in zip(self.data_str, self.data_int):
                d[k] = v
            return d

        def py_dict_comprehension():
            return {k: v for k, v in zip(self.data_str, self.data_int)}

        def py_dict_constructor():
            return dict(zip(self.data_str, self.data_int))

        def spork_transient_map():
            t = EMPTY_MAP.transient()
            for k, v in zip(self.data_str, self.data_int):
                t.assoc_mut(k, v)
            return t.persistent()

        def spork_persistent_map():
            m = EMPTY_MAP
            for k, v in zip(self.data_str, self.data_int):
                m = m.assoc(k, v)
            return m

        def spork_hash_map_factory():
            # Flatten to k1, v1, k2, v2, ...
            args = []
            for k, v in zip(self.data_str, self.data_int):
                args.extend([k, v])
            return hash_map(*args)

        py_build_time = run_benchmark("dict[] loop", py_dict_build, self.ITERS)
        py_comp_time = run_benchmark(
            "dict comprehension", py_dict_comprehension, self.ITERS
        )
        py_ctor_time = run_benchmark("dict(zip(...))", py_dict_constructor, self.ITERS)
        spork_t_time = run_benchmark("TransientMap", spork_transient_map, self.ITERS)
        spork_p_time = run_benchmark(
            "Map.assoc chain", spork_persistent_map, max(1, self.ITERS // 10)
        )
        spork_factory_time = run_benchmark(
            "hash_map(*args)", spork_hash_map_factory, self.ITERS
        )

        print_group(
            f"Map Construction (N={self.N})",
            [
                ("Python dict(zip(k, v))", py_ctor_time),
                ("Python {k: v for ...}", py_comp_time),
                ("Python dict[] loop", py_build_time),
                ("spork-pds hash_map(*args)", spork_factory_time),
                ("spork-pds TransientMap", spork_t_time),
                ("spork-pds Map.assoc() chain", spork_p_time),
            ],
        )

    def bench_map_lookup(self):
        limit = min(self.N, 10000)
        keys_to_lookup = self.data_str[:limit]
        missing_keys = [f"missing_{i}" for i in range(limit)]

        def read_py_dict():
            s = 0
            for k in keys_to_lookup:
                s += self.py_dict[k]
            return s

        def read_spork_map():
            s = 0
            for k in keys_to_lookup:
                s += self.spork_map.get(k, 0)
            return s

        def read_py_dict_missing():
            s = 0
            for k in missing_keys:
                s += self.py_dict.get(k, 0)
            return s

        def read_spork_map_missing():
            s = 0
            for k in missing_keys:
                s += self.spork_map.get(k, 0)
            return s

        py_time = run_benchmark("dict[]", read_py_dict, self.ITERS)
        spork_time = run_benchmark("Map.get", read_spork_map, self.ITERS)
        py_miss_time = run_benchmark(
            "dict.get(missing)", read_py_dict_missing, self.ITERS
        )
        spork_miss_time = run_benchmark(
            "Map.get(missing)", read_spork_map_missing, self.ITERS
        )

        print_group(
            f"Map Lookup ({limit} lookups)",
            [
                ("Python dict[k]", py_time),
                ("spork-pds Map.get(k)", spork_time),
                ("Python dict.get(missing)", py_miss_time),
                ("spork-pds Map.get(missing)", spork_miss_time),
            ],
        )

    def bench_map_dissoc(self):
        dissoc_count = min(1000, self.N // 10)
        keys_to_remove = self.data_str[:dissoc_count]

        def py_dict_del():
            d = self.py_dict.copy()
            for k in keys_to_remove:
                del d[k]
            return d

        def spork_transient_dissoc():
            t = self.spork_map.transient()
            for k in keys_to_remove:
                t.dissoc_mut(k)
            return t.persistent()

        def spork_persistent_dissoc():
            m = self.spork_map
            for k in keys_to_remove:
                m = m.dissoc(k)
            return m

        py_time = run_benchmark("dict copy+del", py_dict_del, self.ITERS)
        spork_t_time = run_benchmark(
            "Transient.dissoc_mut", spork_transient_dissoc, self.ITERS
        )
        spork_p_time = run_benchmark("Map.dissoc", spork_persistent_dissoc, self.ITERS)

        print_group(
            f"Map Dissoc ({dissoc_count} removals)",
            [
                ("Python dict copy+del", py_time),
                ("spork-pds Transient.dissoc_mut()", spork_t_time),
                ("spork-pds Map.dissoc()", spork_p_time),
            ],
        )

    def bench_map_iteration(self):
        def iter_py_dict_keys():
            s = 0
            for k in self.py_dict.keys():
                s += len(k)
            return s

        def iter_py_dict_values():
            s = 0
            for v in self.py_dict.values():
                s += v
            return s

        def iter_py_dict_items():
            s = 0
            for k, v in self.py_dict.items():
                s += v
            return s

        def iter_spork_map_keys():
            s = 0
            for k in self.spork_map.keys():
                s += len(k)
            return s

        def iter_spork_map_values():
            s = 0
            for v in self.spork_map.values():
                s += v
            return s

        def iter_spork_map_items():
            s = 0
            for k, v in self.spork_map.items():
                s += v
            return s

        py_keys_time = run_benchmark("dict.keys()", iter_py_dict_keys, self.ITERS)
        spork_keys_time = run_benchmark("Map.keys()", iter_spork_map_keys, self.ITERS)
        py_vals_time = run_benchmark("dict.values()", iter_py_dict_values, self.ITERS)
        spork_vals_time = run_benchmark(
            "Map.values()", iter_spork_map_values, self.ITERS
        )
        py_items_time = run_benchmark("dict.items()", iter_py_dict_items, self.ITERS)
        spork_items_time = run_benchmark(
            "Map.items()", iter_spork_map_items, self.ITERS
        )

        print_group(
            "Map Iteration - keys()",
            [
                ("Python dict.keys()", py_keys_time),
                ("spork-pds Map.keys()", spork_keys_time),
            ],
        )

        print_group(
            "Map Iteration - values()",
            [
                ("Python dict.values()", py_vals_time),
                ("spork-pds Map.values()", spork_vals_time),
            ],
        )

        print_group(
            "Map Iteration - items()",
            [
                ("Python dict.items()", py_items_time),
                ("spork-pds Map.items()", spork_items_time),
            ],
        )

    # --- Set Operations ---

    def bench_set_construction(self):
        def py_set_build():
            s = set()
            for i in self.data_int:
                s.add(i)
            return s

        def py_set_comprehension():
            return {i for i in self.data_int}

        def py_set_constructor():
            return set(self.data_int)

        def spork_transient_set():
            t = EMPTY_SET.transient()
            for i in self.data_int:
                t.conj_mut(i)
            return t.persistent()

        def spork_persistent_set():
            s = EMPTY_SET
            for i in self.data_int:
                s = s.conj(i)
            return s

        py_build_time = run_benchmark("set.add loop", py_set_build, self.ITERS)
        py_comp_time = run_benchmark(
            "set comprehension", py_set_comprehension, self.ITERS
        )
        py_ctor_time = run_benchmark("set(iterable)", py_set_constructor, self.ITERS)
        spork_t_time = run_benchmark("TransientSet", spork_transient_set, self.ITERS)
        spork_p_time = run_benchmark(
            "Set.conj chain", spork_persistent_set, max(1, self.ITERS // 10)
        )

        print_group(
            f"Set Construction (N={self.N})",
            [
                ("Python set(iterable)", py_ctor_time),
                ("Python {x for x in ...}", py_comp_time),
                ("Python set.add() loop", py_build_time),
                ("spork-pds TransientSet", spork_t_time),
                ("spork-pds Set.conj() chain", spork_p_time),
            ],
        )

    def bench_set_membership(self):
        num_checks = min(10000, self.N)
        check_values = [random.randint(0, self.N * 2) for _ in range(num_checks)]

        def py_set_contains():
            c = 0
            for v in check_values:
                if v in self.py_set:
                    c += 1
            return c

        def spork_set_contains():
            c = 0
            for v in check_values:
                if v in self.spork_set:
                    c += 1
            return c

        py_time = run_benchmark("set.__contains__", py_set_contains, self.ITERS)
        spork_time = run_benchmark("Set.__contains__", spork_set_contains, self.ITERS)

        print_group(
            f"Set Membership ({num_checks} lookups)",
            [
                ("Python set (in)", py_time),
                ("spork-pds Set (in)", spork_time),
            ],
        )

    def bench_set_disj(self):
        disj_count = min(1000, self.N // 10)
        elements_to_remove = self.data_int[:disj_count]

        def py_set_discard():
            s = self.py_set.copy()
            for e in elements_to_remove:
                s.discard(e)
            return s

        def spork_transient_disj():
            t = self.spork_set.transient()
            for e in elements_to_remove:
                t.disj_mut(e)
            return t.persistent()

        def spork_persistent_disj():
            s = self.spork_set
            for e in elements_to_remove:
                s = s.disj(e)
            return s

        py_time = run_benchmark("set.discard", py_set_discard, self.ITERS)
        spork_t_time = run_benchmark(
            "Transient.disj_mut", spork_transient_disj, self.ITERS
        )
        spork_p_time = run_benchmark("Set.disj", spork_persistent_disj, self.ITERS)

        print_group(
            f"Set Disj ({disj_count} removals)",
            [
                ("Python set copy+discard", py_time),
                ("spork-pds Transient.disj_mut()", spork_t_time),
                ("spork-pds Set.disj()", spork_p_time),
            ],
        )

    def bench_set_iteration(self):
        def iter_py_set():
            s = 0
            for x in self.py_set:
                s += x
            return s

        def iter_spork_set():
            s = 0
            for x in self.spork_set:
                s += x
            return s

        py_time = run_benchmark("set iteration", iter_py_set, self.ITERS)
        spork_time = run_benchmark("Set iteration", iter_spork_set, self.ITERS)

        print_group(
            "Set Iteration",
            [
                ("Python set", py_time),
                ("spork-pds Set", spork_time),
            ],
        )

    # --- Structural Sharing ---

    def bench_structural_sharing(self):
        idx = self.N // 2
        val = 99999
        key = self.data_str[self.N // 2]
        new_elem = self.N + 1000

        def py_list_copy_modify():
            new_l = self.py_list.copy()
            new_l[idx] = val
            return new_l

        def spork_vec_assoc():
            return self.spork_vec.assoc(idx, val)

        def py_dict_copy_modify():
            new_d = self.py_dict.copy()
            new_d[key] = val
            return new_d

        def spork_map_assoc():
            return self.spork_map.assoc(key, val)

        def py_set_copy_add():
            new_s = self.py_set.copy()
            new_s.add(new_elem)
            return new_s

        def spork_set_conj():
            return self.spork_set.conj(new_elem)

        py_vec_time = run_benchmark(
            "list.copy()+mod", py_list_copy_modify, max(5, self.ITERS // 5)
        )
        spork_vec_time = run_benchmark("Vector.assoc", spork_vec_assoc, self.ITERS)

        py_map_time = run_benchmark(
            "dict.copy()+mod", py_dict_copy_modify, max(5, self.ITERS // 5)
        )
        spork_map_time = run_benchmark("Map.assoc", spork_map_assoc, self.ITERS)

        py_set_time = run_benchmark(
            "set.copy()+add", py_set_copy_add, max(5, self.ITERS // 5)
        )
        spork_set_time = run_benchmark("Set.conj", spork_set_conj, self.ITERS)

        print(
            f"  {Colors.BLUE}Scenario: Single update on collection of size {self.N}{Colors.ENDC}"
        )
        print()

        print_group(
            "Vector: Single Element Update",
            [
                ("Python list.copy() + modify", py_vec_time),
                ("spork-pds Vector.assoc()", spork_vec_time),
            ],
        )

        print_group(
            "Map: Single Key Update",
            [
                ("Python dict.copy() + modify", py_map_time),
                ("spork-pds Map.assoc()", spork_map_time),
            ],
        )

        print_group(
            "Set: Single Element Add",
            [
                ("Python set.copy() + add", py_set_time),
                ("spork-pds Set.conj()", spork_set_time),
            ],
        )

    def bench_multiple_updates(self):
        update_count = min(100, self.N // 10)
        indices = [random.randint(0, self.N - 1) for _ in range(update_count)]
        keys = [
            self.data_str[random.randint(0, self.N - 1)] for _ in range(update_count)
        ]

        def py_list_copy_multi():
            l = self.py_list
            for idx in indices:
                l = l.copy()
                l[idx] = 99999
            return l

        def spork_vec_assoc_multi():
            v = self.spork_vec
            for idx in indices:
                v = v.assoc(idx, 99999)
            return v

        def spork_vec_transient_multi():
            t = self.spork_vec.transient()
            for idx in indices:
                t.assoc_mut(idx, 99999)
            return t.persistent()

        def py_dict_copy_multi():
            d = self.py_dict
            for k in keys:
                d = d.copy()
                d[k] = 99999
            return d

        def spork_map_assoc_multi():
            m = self.spork_map
            for k in keys:
                m = m.assoc(k, 99999)
            return m

        def spork_map_transient_multi():
            t = self.spork_map.transient()
            for k in keys:
                t.assoc_mut(k, 99999)
            return t.persistent()

        py_vec_time = run_benchmark(
            "list copy chain", py_list_copy_multi, max(5, self.ITERS // 5)
        )
        spork_vec_p_time = run_benchmark(
            "Vector.assoc chain", spork_vec_assoc_multi, self.ITERS
        )
        spork_vec_t_time = run_benchmark(
            "Transient.assoc_mut", spork_vec_transient_multi, self.ITERS
        )

        py_map_time = run_benchmark(
            "dict copy chain", py_dict_copy_multi, max(5, self.ITERS // 5)
        )
        spork_map_p_time = run_benchmark(
            "Map.assoc chain", spork_map_assoc_multi, self.ITERS
        )
        spork_map_t_time = run_benchmark(
            "Transient.assoc_mut", spork_map_transient_multi, self.ITERS
        )

        print(
            f"  {Colors.BLUE}Scenario: {update_count} updates on collection of size {self.N}{Colors.ENDC}"
        )
        print()

        print_group(
            "Vector: Multiple Updates",
            [
                ("Python list copy chain", py_vec_time),
                ("spork-pds Vector.assoc() chain", spork_vec_p_time),
                ("spork-pds Transient.assoc_mut()", spork_vec_t_time),
            ],
        )

        print_group(
            "Map: Multiple Updates",
            [
                ("Python dict copy chain", py_map_time),
                ("spork-pds Map.assoc() chain", spork_map_p_time),
                ("spork-pds Transient.assoc_mut()", spork_map_t_time),
            ],
        )

    # --- Conversion / Interop ---

    def bench_sequences(self):
        def py_list_to_iter():
            return list(iter(self.py_list))

        def spork_vec_to_seq():
            return list(self.spork_vec.to_seq())

        def spork_map_to_seq():
            return list(self.spork_map.to_seq())

        def spork_set_to_seq():
            return list(self.spork_set.to_seq())

        py_time = run_benchmark("list(iter(list))", py_list_to_iter, self.ITERS)
        vec_time = run_benchmark("Vector.to_seq", spork_vec_to_seq, self.ITERS)
        map_time = run_benchmark("Map.to_seq", spork_map_to_seq, self.ITERS)
        set_time = run_benchmark("Set.to_seq", spork_set_to_seq, self.ITERS)

        print_group(
            "Eager Sequence Conversion (to Cons list)",
            [
                ("Python list(iter(list))", py_time),
                ("spork-pds Vector.to_seq()", vec_time),
                ("spork-pds Map.to_seq()", map_time),
                ("spork-pds Set.to_seq()", set_time),
            ],
        )


    def bench_len(self):
        def py_list_len():
            return len(self.py_list)

        def py_dict_len():
            return len(self.py_dict)

        def py_set_len():
            return len(self.py_set)

        def spork_vec_len():
            return len(self.spork_vec)

        def spork_map_len():
            return len(self.spork_map)

        def spork_set_len():
            return len(self.spork_set)

        py_list_time = run_benchmark("len(list)", py_list_len, self.ITERS)
        spork_vec_time = run_benchmark("len(Vector)", spork_vec_len, self.ITERS)
        py_dict_time = run_benchmark("len(dict)", py_dict_len, self.ITERS)
        spork_map_time = run_benchmark("len(Map)", spork_map_len, self.ITERS)
        py_set_time = run_benchmark("len(set)", py_set_len, self.ITERS)
        spork_set_time = run_benchmark("len(Set)", spork_set_len, self.ITERS)

        print_group(
            "Length Operation",
            [
                ("Python len(list)", py_list_time),
                ("spork-pds len(Vector)", spork_vec_time),
                ("Python len(dict)", py_dict_time),
                ("spork-pds len(Map)", spork_map_time),
                ("Python len(set)", py_set_time),
                ("spork-pds len(Set)", spork_set_time),
            ],
        )

    # --- NumPy Interop ---

    def bench_numpy_interop(self):
        if not HAS_NUMPY:
            print(f"{Colors.YELLOW}NumPy not installed, skipping.{Colors.ENDC}")
            return

        def numpy_from_list():
            return np.array(self.data_float)

        def numpy_from_spork():
            return np.array(self.spork_dvec, copy=False)

        np_from_list = np.array(self.data_float)
        np_from_spork = np.array(self.spork_dvec, copy=False)

        def numpy_sum_list():
            return np.sum(np_from_list)

        def numpy_sum_spork():
            return np.sum(np_from_spork)

        def numpy_mean_list():
            return np.mean(np_from_list)

        def numpy_mean_spork():
            return np.mean(np_from_spork)

        py_time = run_benchmark("np.array(list)", numpy_from_list, self.ITERS)
        spork_time = run_benchmark("np.array(DoubleVec)", numpy_from_spork, self.ITERS)

        sum_list_time = run_benchmark("np.sum(from list)", numpy_sum_list, self.ITERS)
        sum_spork_time = run_benchmark(
            "np.sum(from DoubleVec)", numpy_sum_spork, self.ITERS
        )
        mean_list_time = run_benchmark(
            "np.mean(from list)", numpy_mean_list, self.ITERS
        )
        mean_spork_time = run_benchmark(
            "np.mean(from DoubleVec)", numpy_mean_spork, self.ITERS
        )

        print_group(
            "NumPy Array Creation",
            [
                ("np.array(Python list)", py_time),
                ("np.array(DoubleVector) [zero-copy]", spork_time),
            ],
        )

        print_group(
            "NumPy Operations",
            [
                ("np.sum(from list)", sum_list_time),
                ("np.sum(from DoubleVector)", sum_spork_time),
                ("np.mean(from list)", mean_list_time),
                ("np.mean(from DoubleVector)", mean_spork_time),
            ],
        )

        # Verify it works
        arr = numpy_from_spork()
        print(f"  {Colors.BLUE}Verification:{Colors.ENDC} Array sum={arr.sum():.2f}")


def main():
    parser = argparse.ArgumentParser(description="spork-pds benchmark suite")
    parser.add_argument(
        "--size",
        type=positive_int,
        default=100000,
        help="Number of elements in collections (default: 100000)",
    )
    parser.add_argument(
        "--iter",
        type=positive_int,
        default=50,
        help="Number of iterations for timing (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generated workloads (default: 0)",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    print(f"{Colors.BOLD}spork-pds Performance Benchmark{Colors.ENDC}")
    print(f"Size: {args.size}, Iterations: {args.iter}, Seed: {args.seed}")
    print("-" * 60)

    b = Benchmarks(args.size, args.iter)

    # Vector benchmarks
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  VECTOR BENCHMARKS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")

    b.bench_vector_construction()
    b.bench_typed_vector_construction()
    b.bench_vector_access()
    b.bench_vector_pop()

    # Map benchmarks
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  MAP BENCHMARKS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")

    b.bench_map_construction()
    b.bench_map_lookup()
    b.bench_map_dissoc()
    b.bench_map_iteration()

    # Set benchmarks
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  SET BENCHMARKS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")

    b.bench_set_construction()
    b.bench_set_membership()
    b.bench_set_disj()
    b.bench_set_iteration()

    # Structural sharing benchmarks
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  STRUCTURAL SHARING BENCHMARKS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")

    b.bench_structural_sharing()
    b.bench_multiple_updates()

    # Utility benchmarks
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}  UTILITY BENCHMARKS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")

    b.bench_len()
    b.bench_sequences()

    # NumPy interop
    if HAS_NUMPY:
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.HEADER}  NUMPY INTEROP BENCHMARKS{Colors.ENDC}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")

        b.bench_numpy_interop()

    print(f"\n{Colors.BOLD}Benchmark complete!{Colors.ENDC}")


if __name__ == "__main__":
    main()
