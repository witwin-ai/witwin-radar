"""Legacy entry point for native Dirichlet CUDA kernel benchmarking."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from benchmark_dirichlet_cuda import main


if __name__ == "__main__":
    main()
