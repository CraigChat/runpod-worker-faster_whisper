import importlib.util
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np

from faster_whisper.audio import decode_audio

logger = logging.getLogger("runpod-worker")

SAMPLING_RATE = 16000

# faster_whisper/audio.py only imports av and numpy. Loading it by path keeps the child
# from importing the rest of the package
_AUDIO_PY = os.path.join(
    importlib.util.find_spec("faster_whisper").submodule_search_locations[0], "audio.py"
)
_CHILD = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("fw_audio", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
audio = mod.decode_audio(sys.argv[2], sampling_rate=int(sys.argv[3]))
# decode_audio returns s16 / 32768, so this round-trips exactly
sys.stdout.buffer.write((audio * 32768.0).astype("int16").tobytes())
"""


def available_cores() -> int:
    """CPUs this process may use. Containers (RunPod) cap CPU with a cgroup quota while
    affinity still lists every host core, so the quota wins when present."""
    if os.getenv("PREP_THREADS"):
        return max(1, int(os.environ["PREP_THREADS"]))
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for read in (
        lambda: open("/sys/fs/cgroup/cpu.max").read().split(),  # cgroup v2
        lambda: (
            open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read().strip(),  # cgroup v1
            open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read().strip(),
        ),
    ):
        try:
            quota, period = read()
            if quota not in ("max", "-1"):
                n = min(n, max(1, int(int(quota) / int(period))))
            break
        except (OSError, ValueError):
            continue
    return max(1, n)


def decode_file(path: str) -> np.ndarray:
    """Decode one file to 16 kHz mono float32 in a child process."""
    try:
        out = subprocess.run(
            [sys.executable, "-c", _CHILD, _AUDIO_PY, path, str(SAMPLING_RATE)],
            capture_output=True,
            check=True,
        ).stdout
        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    except (OSError, subprocess.CalledProcessError) as e:
        stderr = getattr(e, "stderr", b"") or b""
        logger.warning(
            "Subprocess decode failed for %s (%s); decoding in-process. %s",
            path, e, stderr.decode(errors="replace")[-500:],
        )
        return decode_audio(path, sampling_rate=SAMPLING_RATE)


def decode_files(paths: List[str]) -> List[np.ndarray]:
    """Decode several files in parallel, one child process per file."""
    if len(paths) <= 1:
        return [decode_file(p) for p in paths]
    with ThreadPoolExecutor(min(len(paths), available_cores())) as ex:
        return list(ex.map(decode_file, paths))
