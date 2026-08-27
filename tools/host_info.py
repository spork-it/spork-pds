#!/usr/bin/env python3
"""
Dump host system information for benchmark documentation.
"""

import os
import platform
import subprocess
import sys


def get_cpu_info():
    """Get CPU model name."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown"


def get_memory_gb():
    """Get total system memory in GB."""
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / 1024 / 1024, 1)
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            return round(int(result.stdout.strip()) / 1024 / 1024 / 1024, 1)
    except Exception:
        pass
    return "Unknown"


def get_python_impl():
    """Get Python implementation details."""
    impl = platform.python_implementation()
    if impl == "CPython":
        return f"CPython {platform.python_version()}"
    elif impl == "PyPy":
        return f"PyPy {platform.python_version()} ({'.'.join(map(str, sys.pypy_version_info[:3]))})"
    return f"{impl} {platform.python_version()}"


def main():
    print("## Host Information\n")
    print(f"- **OS**: {platform.system()} {platform.release()}")
    print(f"- **CPU**: {get_cpu_info()}")
    print(f"- **Cores**: {os.cpu_count()}")
    print(f"- **Memory**: {get_memory_gb()} GB")
    print(f"- **Python**: {get_python_impl()}")
    print(f"- **Architecture**: {platform.machine()}")


if __name__ == "__main__":
    main()
