"""Nsight Systems capture of paired, warmed motion-to-ADC frames.

Run under nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
--capture-range=cudaProfilerApi. Profiling excludes setup and warmup; clean
latency and correctness measurements belong to benchmark_motion_adc.py.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import benchmark_motion_adc as benchmark
import benchmark_smpl_walk as walk
import torch

from witwin.radar import simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--motion", type=Path, default=Path("tools/fixtures/smpl_walk_genesis_seed10.npz"))
    parser.add_argument("--model-root", type=Path, default=Path("output/smpl-walk/models"))
    args = parser.parse_args()
    body = walk.WalkingBody(args.motion, args.model_root, 128, 10)
    session = walk.make_session(body, {**walk.WALK_RADAR, "output_domain": "beat"})
    before = benchmark.load_snapshot("witwin.radar._profile_before", args.baseline / "simulation.py")
    echoes = {"baseline": before._adaptive_echo, "current": simulation._adaptive_echo}

    def run(name):
        original = echoes[name]

        def echo(*args, **kwargs):
            with torch.cuda.nvtx.range(f"{name}/adaptive_echo"):
                return original(*args, **kwargs)

        with patch.object(simulation, "_adaptive_echo", echo):
            with torch.cuda.nvtx.range(f"{name}/motion_to_adc"):
                return walk.completed(lambda: walk.simulate(*session, 2.3))

    for name in echoes:
        run(name)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    cubes, elapsed = {}, {}
    for name in echoes:
        result, elapsed[name] = run(name)
        cubes[name] = result.cube
    torch.cuda.cudart().cudaProfilerStop()
    assert torch.equal(cubes["baseline"], cubes["current"])
    print(json.dumps({"instrumented_seconds": elapsed, "adc_bit_equal": True}))


if __name__ == "__main__":
    main()
