"""Smoke-test an installed spork-pds distribution.

For a free-threaded interpreter the parent checks default startup behavior and
then launches a second process with ``PYTHON_GIL=0`` explicitly.
"""

import argparse
import os
import subprocess
import sys
import sysconfig
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def smoke_import_and_operations():
    gil_before = getattr(sys, "_is_gil_enabled", lambda: True)()

    import spork.pds as pds
    import spork_pds

    assert pds.Vector is spork_pds.Vector
    assert pds.EMPTY_VECTOR is spork_pds.EMPTY_VECTOR
    assert pds.EMPTY_MAP is spork_pds.EMPTY_MAP
    assert pds.EMPTY_SET is spork_pds.EMPTY_SET

    vector = pds.Vector(range(128))
    mapping = pds.Map((value, value * 2) for value in range(128))
    set_value = pds.Set(range(128))
    sorted_value = pds.SortedVector(reversed(range(128)))
    doubles = pds.DoubleVector(range(128))
    integers = pds.IntVector(range(128))

    assert vector.assoc(64, -1)[64] == -1
    assert mapping.assoc(64, -1)[64] == -1
    assert 128 in set_value.conj(128)
    assert list(sorted_value) == list(range(128))
    assert memoryview(doubles)[-1] == 127.0
    assert memoryview(integers)[-1] == 127
    hashes = hash(vector), hash(mapping), hash(set_value)
    assert (hash(vector), hash(mapping), hash(set_value)) == hashes

    def worker(worker_id):
        builder = vector.transient()
        builder.conj_mut(worker_id)
        return (
            vector[64],
            mapping[64],
            64 in set_value,
            builder.persistent()[-1],
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(4)))
    assert results == [(64, 128, True, worker_id) for worker_id in range(4)]

    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    gil_after = getattr(sys, "_is_gil_enabled", lambda: True)()
    if free_threaded:
        assert not gil_before, "GIL enabled before importing installed wheel"
        assert not gil_after, "installed wheel enabled the GIL"
        abi_marker = f"{sys.version_info.major}{sys.version_info.minor}t"
        assert abi_marker in sysconfig.get_config_var("SOABI")
        assert abi_marker in sysconfig.get_config_var("EXT_SUFFIX")
        assert abi_marker in Path(spork_pds.__file__).name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-no-gil-child", action="store_true")
    args = parser.parse_args()

    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    smoke_import_and_operations()

    if free_threaded and not args.explicit_no_gil_child:
        env = os.environ.copy()
        env["PYTHON_GIL"] = "0"
        env["PYTHONWARNINGS"] = "error::RuntimeWarning"
        sanitizer_runtime = env.get("SPORK_PDS_DYLD_INSERT_LIBRARIES")
        if sanitizer_runtime:
            env["DYLD_INSERT_LIBRARIES"] = sanitizer_runtime
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--explicit-no-gil-child",
                ],
                cwd=directory,
                env=env,
                check=True,
            )

    mode = "free-threaded no-GIL" if free_threaded else "regular-GIL"
    print(f"installed-distribution smoke test passed ({mode})")


if __name__ == "__main__":
    main()
