import hashlib
import json
import os
import platform
from pathlib import Path

import psutil
import torch


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resource_snapshot(device):
    process = psutil.Process()
    resources = {
        "os": platform.platform(), "python": platform.python_version(), "torch": str(torch.__version__),
        "cpu": os.getenv("PROCESSOR_IDENTIFIER") or platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 2), "device": str(device),
        "torch_threads": torch.get_num_threads(), "process_rss_mib": round(process.memory_info().rss / 2**20, 2),
    }
    if device.type == "cuda":
        resources["gpu"] = torch.cuda.get_device_name(device)
        resources["peak_cuda_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
    if hasattr(process.memory_info(), "peak_wset"):
        resources["peak_working_set_mib"] = round(process.memory_info().peak_wset / 2**20, 2)
    return resources


def image_on_device(image, device):
    return image.to(device=device, memory_format=torch.channels_last)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def read_checkpoint(path, device, fingerprint, stage):
    state = torch.load(path, map_location=device, weights_only=True)
    if state["manifest"] != fingerprint or state["stage"] != stage:
        raise ValueError(f"Checkpoint stage or dataset differs: {path}")
    return state
