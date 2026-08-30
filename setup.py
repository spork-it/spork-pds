import os
import sys

from setuptools import Extension, setup


# Check for a debug build (used by `make build-debug`).
if os.environ.get("DEBUG_BUILD"):
    if sys.platform != "win32":
        extra_compile_args = ["-O1", "-g"]
    else:
        extra_compile_args = ["/Od", "/Zi"]
else:
    # Release build with optimization and security hardening.
    if sys.platform != "win32":
        extra_compile_args = [
            "-O3",
            "-D_FORTIFY_SOURCE=2",
            "-fstack-protector-strong",
        ]
    else:
        extra_compile_args = [
            "/O2",
            "/GS",
        ]


pds_extension = Extension(
    "spork_pds",
    sources=[
        "src/common.c",
        "src/cons.c",
        "src/vector.c",
        "src/double_vector.c",
        "src/int_vector.c",
        "src/hamt.c",
        "src/map.c",
        "src/set.c",
        "src/sorted_vector.c",
        "src/module.c",
    ],
    depends=["src/pds_internal.h"],
    extra_compile_args=extra_compile_args,
)

setup(ext_modules=[pds_extension])
