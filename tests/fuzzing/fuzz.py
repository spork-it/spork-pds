#!/usr/bin/env python3
"""Base fuzzing framework for spork-pds.

This module provides a base class for fuzz tests and a runner to execute them.

Usage:
    python -m tests.fuzzing.fuzz [--examples N] [--steps N] [--seed N] [pattern...]

Example:
    .venv/bin/python -m tests.fuzzing                    # Run all fuzz tests
    .venv/bin/python -m tests.fuzzing vector             # Run tests matching 'vector'
    .venv/bin/python -m tests.fuzzing --examples 5000    # Run with custom params
"""

import abc
import argparse
import gc
import importlib
import random
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def get_mem_mb() -> float:
    """Get current peak RSS in MiB."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return max_rss / divisor


def force_gc() -> float:
    """Force full garbage collection and return memory after."""
    gc.collect()
    gc.collect()
    gc.collect()
    return get_mem_mb()


def random_value() -> Any:
    """Generate a random hashable value."""
    choice = random.randint(0, 5)
    if choice == 0:
        return random.randint(-10000, 10000)
    elif choice == 1:
        return random.random() * 1000
    elif choice == 2:
        return "".join(random.choices("abcdefghij", k=random.randint(0, 20)))
    elif choice == 3:
        return None
    elif choice == 4:
        return random.choice([True, False])
    else:
        return (random.randint(0, 100), random.randint(0, 100))


@dataclass
class MemoryStats:
    """Statistics about memory behavior during fuzzing."""

    initial_mem: float = 0.0
    warmup_mem: float = 0.0
    final_mem: float = 0.0
    peak_mem: float = 0.0

    # Steady-state tracking
    steady_state_samples: list[tuple[int, float]] = field(default_factory=list)

    # Cleanup verification
    cleanup_checks: list[tuple[float, float, float]] = field(
        default_factory=list
    )  # (before, after_release, after_gc)

    def warmup_overhead(self) -> float:
        """Memory allocated during warmup (globals, caches, etc.)."""
        return self.warmup_mem - self.initial_mem

    def steady_state_growth(self) -> float:
        """Memory growth during steady-state phase."""
        return self.final_mem - self.warmup_mem

    def total_growth(self) -> float:
        """Total memory growth."""
        return self.final_mem - self.initial_mem

    def growth_rate_per_1k_ops(self) -> float | None:
        """Calculate memory growth rate per 1000 operations during steady-state.

        Returns None if insufficient data points.
        A truly leaking implementation will have a positive, consistent rate.
        A healthy implementation will have a rate near zero.
        """
        if len(self.steady_state_samples) < 3:
            return None

        # Use linear regression to find the growth rate
        samples = self.steady_state_samples
        n = len(samples)
        sum_x = sum(ops for ops, _ in samples)
        sum_y = sum(mem for _, mem in samples)
        sum_xy = sum(ops * mem for ops, mem in samples)
        sum_xx = sum(ops * ops for ops, _ in samples)

        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        # Convert to per-1000-ops rate
        return slope * 1000

    def cleanup_efficiency(self) -> float | None:
        """Calculate how well memory is freed during cleanup checks.

        Returns the average percentage of memory recovered after GC.
        100% means all allocated memory was freed.
        Returns None if no cleanup checks were performed.
        """
        if not self.cleanup_checks:
            return None

        efficiencies = []
        for before, after_release, after_gc in self.cleanup_checks:
            allocated = after_release - before
            if allocated > 1.0:  # Only count if significant allocation
                recovered = after_release - after_gc
                efficiency = (recovered / allocated) * 100 if allocated > 0 else 100
                efficiencies.append(min(100, max(0, efficiency)))

        return sum(efficiencies) / len(efficiencies) if efficiencies else None


class Fuzzer(abc.ABC):
    """Base class for fuzz tests.

    Subclasses must implement:
        - name: class attribute with the fuzzer name
        - reset(): reset state for a new example
        - do_random_operation(): perform one random operation
        - check_invariants(): verify state is correct

    Optionally override:
        - setup(): called once before running
        - teardown(): called once after running
        - get_stats(): return dict of stats to display
        - release_all(): explicitly release all held references for cleanup checks
    """

    name: str = "unnamed"

    def __init__(self):
        self.operations = 0
        self.op_counts: dict[str, int] = {}

    def record_op(self, name: str):
        """Record that an operation was performed."""
        self.operations += 1
        self.op_counts[name] = self.op_counts.get(name, 0) + 1

    def setup(self):
        """Called once before running. Override if needed."""
        pass

    def teardown(self):
        """Called once after running. Override if needed."""
        pass

    def release_all(self):
        """Release all references to allow GC. Override if needed.

        This is called during cleanup verification checks to ensure
        memory is properly freed when objects are released.
        Default implementation just calls reset().
        """
        self.reset()

    @abc.abstractmethod
    def reset(self):
        """Reset state for a new example."""
        pass

    @abc.abstractmethod
    def do_random_operation(self):
        """Perform one random operation."""
        pass

    @abc.abstractmethod
    def check_invariants(self):
        """Verify that the current state is correct.

        Should raise AssertionError if invariants are violated.
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Return additional stats to display. Override if needed."""
        return {}


@dataclass
class LeakCheckConfig:
    """Configuration for memory leak detection."""

    # Warmup: ignore memory growth during first N% of examples
    # This accounts for global allocations, caches, etc.
    warmup_fraction: float = 0.1

    # Maximum allowed steady-state memory growth rate (MB per 1000 ops)
    # A truly leaking implementation grows linearly with operations
    max_growth_rate_per_1k_ops: float = 0.5

    # Minimum cleanup efficiency (percentage of memory freed after release)
    # If objects are properly reference counted, memory should be freed
    min_cleanup_efficiency: float = 50.0

    # Maximum absolute steady-state growth (MB)
    # Even with zero leak rate, cap absolute growth
    max_steady_state_growth_mb: float = 200.0

    # How often to sample memory for growth rate calculation (in examples)
    sample_interval: int = 50

    # How often to do cleanup verification checks (in examples)
    # Set to 0 to disable
    cleanup_check_interval: int = 200


class FuzzRunner:
    """Runs fuzz tests and reports results."""

    def __init__(
        self,
        examples: int = 1000,
        steps: int = 200,
        seed: int | None = None,
        leak_config: LeakCheckConfig | None = None,
    ):
        self.examples = examples
        self.steps = steps
        self.leak_config = leak_config or LeakCheckConfig()

        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0, 2**32)
        random.seed(self.seed)

    def _do_cleanup_check(self, fuzzer: Fuzzer, mem_stats: MemoryStats) -> None:
        """Perform a cleanup verification check."""
        before = get_mem_mb()
        fuzzer.release_all()
        after_release = get_mem_mb()
        after_gc = force_gc()
        mem_stats.cleanup_checks.append((before, after_release, after_gc))

    def run(self, fuzzer: Fuzzer) -> bool:
        """Run a fuzzer. Returns True if passed, False if failed."""
        print(f"Fuzz: {fuzzer.name}")
        print(f"  Examples: {self.examples:,}")
        print(f"  Steps per example: {self.steps}")
        print(f"  Seed: {self.seed}")
        print()

        # Initialize memory tracking
        mem_stats = MemoryStats()
        mem_stats.initial_mem = force_gc()
        mem_stats.peak_mem = mem_stats.initial_mem

        warmup_examples = int(self.examples * self.leak_config.warmup_fraction)
        in_warmup = True

        start_time = time.time()
        last_print = start_time
        example = 0
        step = 0

        fuzzer.setup()

        try:
            for example in range(self.examples):
                # Transition from warmup to steady-state
                if in_warmup and example >= warmup_examples:
                    in_warmup = False
                    mem_stats.warmup_mem = force_gc()

                fuzzer.reset()

                for step in range(self.steps):
                    fuzzer.do_random_operation()
                    fuzzer.check_invariants()

                # Track peak memory
                current_mem = get_mem_mb()
                mem_stats.peak_mem = max(mem_stats.peak_mem, current_mem)

                # Sample memory for growth rate calculation (steady-state only)
                if (
                    not in_warmup
                    and self.leak_config.sample_interval > 0
                    and example % self.leak_config.sample_interval == 0
                ):
                    mem_after_gc = force_gc()
                    mem_stats.steady_state_samples.append(
                        (fuzzer.operations, mem_after_gc)
                    )

                # Periodic cleanup verification
                if (
                    self.leak_config.cleanup_check_interval > 0
                    and example > 0
                    and example % self.leak_config.cleanup_check_interval == 0
                ):
                    self._do_cleanup_check(fuzzer, mem_stats)

                # Progress output every second
                now = time.time()
                if now - last_print >= 1.0:
                    elapsed = now - start_time
                    rate = (example + 1) / elapsed
                    current_mem = get_mem_mb()
                    delta_mem = current_mem - mem_stats.initial_mem
                    phase = "warmup" if in_warmup else "steady"

                    print(
                        f"[{elapsed:6.1f}s] "
                        f"ex:{example + 1:>6,} | "
                        f"ops:{fuzzer.operations:>8,} | "
                        f"{rate:>5.1f}/s | "
                        f"rss:{current_mem:.0f}MB ({delta_mem:+.0f}MB) [{phase}]"
                    )
                    last_print = now

            # Final memory measurement
            mem_stats.final_mem = force_gc()

            # Final stats
            elapsed = time.time() - start_time

            print()
            print(
                f"Completed {self.examples:,} examples, "
                f"{fuzzer.operations:,} operations in {elapsed:.1f}s"
            )
            print(f"  Operations: {fuzzer.op_counts}")

            # Show custom stats
            custom_stats = fuzzer.get_stats()
            for key, value in custom_stats.items():
                print(f"  {key}: {value}")

            # Memory analysis
            print()
            print("Memory Analysis:")
            print(
                f"  Initial: {mem_stats.initial_mem:.1f}MB | "
                f"After warmup: {mem_stats.warmup_mem:.1f}MB | "
                f"Final: {mem_stats.final_mem:.1f}MB | "
                f"Peak: {mem_stats.peak_mem:.1f}MB"
            )
            print(
                f"  Warmup overhead: {mem_stats.warmup_overhead():+.1f}MB "
                f"(globals, caches, etc. - ignored)"
            )
            print(f"  Steady-state growth: {mem_stats.steady_state_growth():+.1f}MB")

            growth_rate = mem_stats.growth_rate_per_1k_ops()
            if growth_rate is not None:
                print(f"  Growth rate: {growth_rate:+.3f}MB per 1000 ops")

            cleanup_eff = mem_stats.cleanup_efficiency()
            if cleanup_eff is not None:
                print(f"  Cleanup efficiency: {cleanup_eff:.1f}%")

            fuzzer.teardown()

            # Evaluate pass/fail based on multiple criteria
            failures = []

            # Check 1: Growth rate during steady-state
            if growth_rate is not None:
                if growth_rate > self.leak_config.max_growth_rate_per_1k_ops:
                    failures.append(
                        f"Growth rate {growth_rate:.3f}MB/1k ops exceeds limit "
                        f"{self.leak_config.max_growth_rate_per_1k_ops:.3f}MB/1k ops"
                    )

            # Check 2: Absolute steady-state growth
            steady_growth = mem_stats.steady_state_growth()
            if steady_growth > self.leak_config.max_steady_state_growth_mb:
                failures.append(
                    f"Steady-state growth {steady_growth:.1f}MB exceeds limit "
                    f"{self.leak_config.max_steady_state_growth_mb:.1f}MB"
                )

            # Check 3: Cleanup efficiency
            if cleanup_eff is not None:
                if cleanup_eff < self.leak_config.min_cleanup_efficiency:
                    failures.append(
                        f"Cleanup efficiency {cleanup_eff:.1f}% below minimum "
                        f"{self.leak_config.min_cleanup_efficiency:.1f}%"
                    )

            if failures:
                print()
                print("  FAILED - Memory leak detected:")
                for failure in failures:
                    print(f"    - {failure}")
                return False
            else:
                print()
                print("  PASSED")
                return True

        except AssertionError as e:
            print()
            print(f"FAILED at example {example + 1}, step {step + 1}!")
            print(f"  Seed: {self.seed}")
            print(f"  Error: {e}")
            fuzzer.teardown()
            return False

        except KeyboardInterrupt:
            print()
            print(f"Interrupted at example {example + 1}")
            fuzzer.teardown()
            return False


def discover_fuzzers() -> list[type[Fuzzer]]:
    """Discover all Fuzzer subclasses in the fuzzing package."""
    fuzzers = []

    # Get the directory containing this file
    fuzzing_dir = Path(__file__).parent

    # Import all fuzz_*.py modules
    for path in fuzzing_dir.glob("fuzz_*.py"):
        module_name = path.stem
        try:
            module = importlib.import_module(f"tests.fuzzing.{module_name}")

            # Find Fuzzer subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Fuzzer)
                    and attr is not Fuzzer
                ):
                    fuzzers.append(attr)
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")

    return fuzzers


def run_suite(
    examples: int = 1000,
    steps: int = 200,
    seed: int | None = None,
    patterns: list[str] | None = None,
    leak_config: LeakCheckConfig | None = None,
) -> int:
    """Run the fuzz test suite.

    Args:
        examples: Number of examples per fuzzer
        steps: Steps per example
        seed: Random seed (None for random)
        patterns: Optional list of patterns to filter fuzzers by name
        leak_config: Configuration for memory leak detection

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    fuzzers = discover_fuzzers()

    if not fuzzers:
        print("No fuzzers found!")
        return 1

    # Filter by patterns if provided
    if patterns:
        filtered = []
        for fuzzer_cls in fuzzers:
            for pattern in patterns:
                if pattern.lower() in fuzzer_cls.name.lower():
                    filtered.append(fuzzer_cls)
                    break
        fuzzers = filtered

    if not fuzzers:
        print("No fuzzers matched the given patterns!")
        return 1

    print(f"Running {len(fuzzers)} fuzzer(s)")
    print("=" * 60)
    print()

    results = []
    runner = FuzzRunner(
        examples=examples,
        steps=steps,
        seed=seed,
        leak_config=leak_config,
    )

    for fuzzer_cls in fuzzers:
        fuzzer = fuzzer_cls()
        passed = runner.run(fuzzer)
        results.append((fuzzer.name, passed))
        print()
        print("=" * 60)
        print()

        # Force GC between fuzzers
        gc.collect()

    # Summary
    print("Summary")
    print("-" * 40)

    passed = sum(1 for _, p in results if p)
    failed = sum(1 for _, p in results if not p)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print()
    print(f"Passed: {passed}, Failed: {failed}")

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Run fuzz test suite")
    parser.add_argument(
        "--examples",
        "-n",
        type=int,
        default=1000,
        help="Number of examples per fuzzer (default: 1000)",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=200,
        help="Steps per example (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-growth-rate",
        type=float,
        default=0.5,
        help="Max memory growth rate in MB per 1000 ops (default: 0.5)",
    )
    parser.add_argument(
        "--max-steady-growth",
        type=float,
        default=200,
        help="Max absolute steady-state memory growth in MB (default: 200)",
    )
    parser.add_argument(
        "--min-cleanup-efficiency",
        type=float,
        default=50,
        help="Min cleanup efficiency percentage (default: 50)",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.1,
        help="Fraction of examples to use as warmup (default: 0.1)",
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        help="Optional patterns to filter fuzzers by name",
    )
    args = parser.parse_args()

    leak_config = LeakCheckConfig(
        warmup_fraction=args.warmup_fraction,
        max_growth_rate_per_1k_ops=args.max_growth_rate,
        max_steady_state_growth_mb=args.max_steady_growth,
        min_cleanup_efficiency=args.min_cleanup_efficiency,
    )

    sys.exit(
        run_suite(
            examples=args.examples,
            steps=args.steps,
            seed=args.seed,
            patterns=args.patterns if args.patterns else None,
            leak_config=leak_config,
        )
    )


if __name__ == "__main__":
    main()
