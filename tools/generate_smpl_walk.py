"""Generate a fixed, locally retained Genesis walking asset for the SMPL benchmark.

Set WITWIN_GENESIS_DATA_DIR to the licensed local Genesis asset directory.
Generation is a separate one-time cost; benchmark_smpl_walk.py consumes the
saved motion without running diffusion or SMPL fitting again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from witwin.genesis import generate_motion_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/smpl-walk"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = generate_motion_asset(
        args.output.resolve(),
        prompt="a person walks forward at a natural steady pace, swinging both arms",
        motion_length=args.seconds,
        seed=args.seed,
        device=0,
        smplify_iters=20,
        output_name="genesis_walk",
    )
    torch.cuda.synchronize()
    record = {
        **result.to_dict(),
        "wall_seconds": time.perf_counter() - start,
        "seed": args.seed,
        "smplify_iters": 20,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "sha256": hashlib.sha256(Path(result.motion_path).read_bytes()).hexdigest(),
    }
    (args.output / "generation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
