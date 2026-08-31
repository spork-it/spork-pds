import tarfile
import zipfile

import pytest

from tools.verify_distributions import verify_sdist, verify_wheel


def make_wheel(tmp_path, filename, extension_name):
    path = tmp_path / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(extension_name, b"native extension placeholder")
    return path


def test_distribution_verifier_distinguishes_regular_and_free_threaded_wheels(
    tmp_path,
):
    regular = make_wheel(
        tmp_path,
        "spork_pds-0.1.4-cp314-cp314-macosx_11_0_arm64.whl",
        "spork_pds.cpython-314-darwin.so",
    )
    free_threaded = make_wheel(
        tmp_path,
        "spork_pds-0.1.4-cp314-cp314t-macosx_11_0_arm64.whl",
        "spork_pds.cpython-314t-darwin.so",
    )

    assert verify_wheel(regular) == "regular"
    assert verify_wheel(free_threaded) == "free-threaded"


def test_distribution_verifier_rejects_wrong_native_free_threaded_suffix(tmp_path):
    wheel = make_wheel(
        tmp_path,
        "spork_pds-0.1.4-cp314-cp314t-manylinux_2_28_x86_64.whl",
        "spork_pds.cpython-314-x86_64-linux-gnu.so",
    )

    with pytest.raises(AssertionError, match="free-threaded ABI suffix"):
        verify_wheel(wheel)


def test_distribution_verifier_checks_release_tools_in_sdist(tmp_path):
    path = tmp_path / "spork_pds-0.1.4.tar.gz"
    required = (
        "pyproject.toml",
        "setup.py",
        "src/module.c",
        "src/pds_internal.h",
        "spork/pds.py",
        "tools/benchmark_free_threading.py",
        "tools/smoke_installed_distribution.py",
        "tools/stress_free_threading.py",
        "tools/verify_distributions.py",
    )
    source = tmp_path / "placeholder"
    source.write_text("placeholder")
    with tarfile.open(path, "w:gz") as archive:
        for relative_path in required:
            archive.add(source, arcname=f"spork_pds-0.1.4/{relative_path}")

    verify_sdist(path)
