#!/usr/bin/env python3
"""Execute every Python documentation example."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
FENCE_RE = re.compile(r"^```([^\n]*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


@dataclass(frozen=True)
class Example:
    path: Path
    line: int
    language: str
    source: str


def examples() -> list[Example]:
    result: list[Example] = []
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for match in FENCE_RE.finditer(text):
            result.append(
                Example(
                    path=path,
                    line=text.count("\n", 0, match.start()) + 1,
                    language=match.group(1).strip(),
                    source=match.group(2),
                )
            )
    return result


def python_environment() -> dict[str, object]:
    """Return a fresh environment with public APIs and common example fixtures."""
    namespace: dict[str, object] = {"__name__": "__docs_example__"}
    exec("from spork_pds import *", namespace)
    exec(
        """
numbers = Vector([1, 2, 3])
config = Map({"host": "localhost", "port": 8000})
roles = Set(["reader", "writer"])
base = Map({"host": "localhost", "port": 8000})
vector = Vector([1, 2, 3])
tags = Set(["stable", "documented"])
floats = vec_f64(1, 2.5, 3)
original = Map({"count": 1})
changes = {"count": 2}
""",
        namespace,
    )
    return namespace


def verify_python(example: Example) -> None:
    # Prefix newlines so traceback locations point at the Markdown source line.
    source = "\n" * example.line + example.source
    code = compile(source, str(example.path.relative_to(ROOT)), "exec")
    exec(code, python_environment())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    python_count = 0
    failures: list[str] = []

    print("Verifying documentation examples...", flush=True)

    for example in examples():
        location = f"{example.path.relative_to(ROOT)}:{example.line}"
        try:
            if example.language == "python":
                verify_python(example)
                python_count += 1
        except Exception as error:  # report every failing fence in one run
            failures.append(f"{location} [{example.language}]: {error}")

    if failures:
        print(f"\n  ✗ {len(failures)} documentation example(s) failed", file=sys.stderr)
        for failure in failures:
            print(f"    {failure}", file=sys.stderr)
        return 1

    print(f"  ✓ Python examples passed {python_count:>4}")
    print("\n✓ Documentation verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
