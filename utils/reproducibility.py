"""
Reproducibility utilities: global seed setting and environment info logging.
"""

from __future__ import annotations

import os
import random
import platform
from typing import Dict, Any

import numpy as np

try:
    import torch
    import torchvision
    import torchaudio
except Exception:
    torch = None
    torchvision = None
    torchaudio = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)


def get_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    try:
        import sklearn
        info["scikit_learn"] = getattr(sklearn, "__version__", "unknown")
    except Exception:
        info["scikit_learn"] = None
    try:
        import sympy
        info["sympy"] = getattr(sympy, "__version__", "unknown")
    except Exception:
        info["sympy"] = None
    info["numpy"] = np.__version__
    info["torch"] = getattr(torch, "__version__", None) if torch is not None else None
    info["torchvision"] = getattr(torchvision, "__version__", None) if torchvision is not None else None
    info["torchaudio"] = getattr(torchaudio, "__version__", None) if torchaudio is not None else None
    return info