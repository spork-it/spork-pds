"""Generate a reproducible Markdown report from the spork-pds benchmarks.

Examples:
    python tools/generate_benchmark_report.py 25000 50000 100000
    python tools/generate_benchmark_report.py --sizes 25000 50000 100000
    python tools/generate_benchmark_report.py --iter 25 --seed 0 -o results.md
"""

import argparse
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def get_script_dir() -> Path:
    return Path(__file__).parent.resolve()


def get_package_version() -> str:
    try:
        return version("spork-pds")
    except PackageNotFoundError:
        return "development checkout"


def run_host_info() -> str:
    script = get_script_dir() / "host_info.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running host_info.py: {result.stderr}", file=sys.stderr)
        raise SystemExit(1)
    return result.stdout.strip()


def run_benchmark(size: int, iterations: int, seed: int) -> str:
    script = get_script_dir() / "benchmark_pds.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--size",
            str(size),
            "--iter",
            str(iterations),
            "--seed",
            str(seed),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error running benchmark (N={size}):", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(1)
    return result.stdout


def format_size(size: int) -> str:
    return f"{size:,}"


def generate_report(sizes: list[int], iterations: int, seed: int) -> str:
    lines = [
        "# spork-pds Benchmark Results",
        "",
        f"- **spork-pds**: {get_package_version()}",
        f"- **Iterations per timing**: {iterations}",
        f"- **Random seed**: {seed}",
        "",
        run_host_info(),
        "",
        "## Results",
    ]

    for size in sizes:
        lines.extend(
            [
                "",
                f"### N={format_size(size)}",
                "",
                "<details>",
                f"<summary>Benchmark output (N={format_size(size)})</summary>",
                "",
                "```text",
                (
                    "$ .venv/bin/python tools/benchmark_pds.py "
                    f"--size {size} --iter {iterations} --seed {seed}"
                ),
            ]
        )

        print(f"Running benchmark with N={format_size(size)}...", file=sys.stderr)
        lines.append(run_benchmark(size, iterations, seed).rstrip())
        lines.extend(["```", "", "</details>"])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown spork-pds benchmark report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s 25000 50000 100000
    %(prog)s --sizes 25000 50000 100000
    %(prog)s --iter 100 --seed 0 25000 50000
        """,
    )
    parser.add_argument(
        "sizes",
        nargs="*",
        type=positive_int,
        help="N values to benchmark (default: 25000 50000 100000)",
    )
    parser.add_argument(
        "--sizes",
        dest="sizes_flag",
        nargs="+",
        type=positive_int,
        help="Alternative way to specify N values",
    )
    parser.add_argument(
        "--iter",
        type=positive_int,
        default=50,
        help="Number of iterations per timing (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generated workloads (default: 0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    sizes = args.sizes_flag or args.sizes or [25000, 50000, 100000]
    sizes = sorted(set(sizes))

    print(
        f"Generating benchmark report for N={', '.join(map(str, sizes))}...",
        file=sys.stderr,
    )
    report = generate_report(sizes, args.iter, args.seed)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
