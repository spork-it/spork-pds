"""Verify wheel ABI separation and source-distribution contents."""

import argparse
import re
import tarfile
import zipfile
from pathlib import Path


FREE_THREADED_TAG = re.compile(r"^cp\d+t$")
CPYTHON_TAG = re.compile(r"^cp\d+t?$")
NATIVE_SUFFIXES = (".so", ".pyd", ".dll")
SDIST_REQUIRED_SUFFIXES = (
    "/pyproject.toml",
    "/setup.py",
    "/src/module.c",
    "/src/pds_internal.h",
    "/spork/pds.py",
    "/tools/benchmark_free_threading.py",
    "/tools/smoke_installed_distribution.py",
    "/tools/stress_free_threading.py",
    "/tools/verify_distributions.py",
)


def distribution_paths(inputs):
    paths = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.whl")))
            paths.extend(sorted(path.glob("*.tar.gz")))
        elif path.is_file():
            paths.append(path)
        else:
            raise SystemExit(f"distribution path does not exist: {path}")
    if not paths:
        raise SystemExit("no wheel or source distributions found")
    return paths


def wheel_tags(path):
    if not path.name.endswith(".whl"):
        raise AssertionError(f"not a wheel: {path}")
    parts = path.name[:-4].split("-")
    if len(parts) < 5:
        raise AssertionError(f"invalid wheel filename: {path.name}")
    return parts[-3].split("."), parts[-2].split("."), parts[-1].split(".")


def verify_wheel(path):
    python_tags, abi_tags, platform_tags = wheel_tags(path)
    if not python_tags or not all(CPYTHON_TAG.match(tag) for tag in python_tags):
        raise AssertionError(f"wheel has no supported CPython tag: {path.name}")
    free_threaded_abis = [tag for tag in abi_tags if FREE_THREADED_TAG.match(tag)]
    regular_abis = [tag for tag in abi_tags if re.match(r"^cp\d+$", tag)]
    if free_threaded_abis and regular_abis:
        raise AssertionError(f"wheel mixes regular and free-threaded ABIs: {path.name}")
    if not free_threaded_abis and not regular_abis:
        raise AssertionError(f"wheel has no supported CPython ABI: {path.name}")

    with zipfile.ZipFile(path) as archive:
        native_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(NATIVE_SUFFIXES)
            and Path(name).name.startswith("spork_pds.")
        ]
    if len(native_members) != 1:
        raise AssertionError(
            f"expected one spork_pds native extension in {path.name}, "
            f"found {native_members}"
        )

    if free_threaded_abis:
        abi = free_threaded_abis[0]
        regular_interpreter_tag = abi[:-1]
        if not any(
            tag in (regular_interpreter_tag, abi) for tag in python_tags
        ):
            raise AssertionError(
                f"free-threaded interpreter/ABI tags disagree in {path.name}"
            )
        marker = abi[2:]
        if marker not in Path(native_members[0]).name:
            raise AssertionError(
                f"native extension lacks the free-threaded ABI suffix in {path.name}"
            )
        mode = "free-threaded"
    else:
        if any(FREE_THREADED_TAG.match(tag) for tag in python_tags):
            raise AssertionError(
                f"regular ABI wheel has a free-threaded interpreter tag: {path.name}"
            )
        mode = "regular"

    if platform_tags == ["any"]:
        raise AssertionError(f"native wheel is incorrectly platform-independent: {path.name}")
    return mode


def verify_sdist(path):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getnames()
    missing = [
        suffix
        for suffix in SDIST_REQUIRED_SUFFIXES
        if not any(name.endswith(suffix) for name in members)
    ]
    if missing:
        raise AssertionError(f"{path.name} is missing required files: {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="distribution files or directories")
    parser.add_argument("--require-cp314t", action="store_true")
    parser.add_argument("--require-regular", action="store_true")
    parser.add_argument("--require-sdist", action="store_true")
    args = parser.parse_args()

    paths = distribution_paths(args.paths)
    modes = []
    sdist_count = 0
    for path in paths:
        if path.name.endswith(".whl"):
            mode = verify_wheel(path)
            modes.append(mode)
            print(f"verified {mode} wheel: {path.name}")
        elif path.name.endswith(".tar.gz"):
            verify_sdist(path)
            sdist_count += 1
            print(f"verified source distribution: {path.name}")

    if args.require_cp314t and "free-threaded" not in modes:
        raise SystemExit("no free-threaded CPython wheel was found")
    if args.require_regular and "regular" not in modes:
        raise SystemExit("no regular CPython wheel was found")
    if args.require_sdist and not sdist_count:
        raise SystemExit("no source distribution was found")
    print("distribution verification passed")


if __name__ == "__main__":
    main()
