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
    sources=["pds.c"],
    extra_compile_args=extra_compile_args,
)

setup(ext_modules=[pds_extension])
