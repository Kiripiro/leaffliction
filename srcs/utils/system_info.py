from __future__ import annotations

import multiprocessing
import os
import platform
from typing import Dict, Union


def get_cpu_info() -> Dict[str, Union[int, bool, str]]:
    try:
        physical_cores = os.cpu_count()
        available_cores = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else physical_cores
        )
        is_apple_silicon = (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        )
        return {
            "physical_cores": physical_cores,
            "available_cores": available_cores,
            "is_apple_silicon": is_apple_silicon,
            "platform": platform.system(),
            "machine": platform.machine(),
        }
    except Exception:
        return {
            "physical_cores": multiprocessing.cpu_count(),
            "available_cores": multiprocessing.cpu_count(),
            "is_apple_silicon": False,
            "platform": "Unknown",
            "machine": "Unknown",
        }


def get_optimal_worker_count() -> int:
    info = get_cpu_info()
    cores = int(info["available_cores"])
    if cores <= 2:
        return 1
    if cores <= 4:
        return max(1, cores - 1)
    if info["is_apple_silicon"]:
        return min(8, cores)
    return max(1, int(cores * 0.75))


def get_system_info() -> Dict[str, Union[int, bool, str]]:
    cpu = get_cpu_info()
    info = {
        "platform": cpu["platform"],
        "machine": cpu["machine"],
        "processor": platform.processor(),
        "physical_cores": cpu["physical_cores"],
        "available_cores": cpu["available_cores"],
        "cpu_count": os.cpu_count(),
        "optimal_workers": get_optimal_worker_count(),
        "apple_silicon": cpu["is_apple_silicon"],
    }
    return info


__all__ = ["get_system_info", "get_optimal_worker_count", "get_cpu_info"]
