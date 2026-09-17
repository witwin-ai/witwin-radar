"""What a batched topology discovery would be worth, and what blocks it.

An adaptive probe in a world that cannot be certified complete needs its own
topology discovery: that is what a probe IS. In the three-wall fixture those
discoveries are 69% of the frame, spent across two Channel calls per probe.

This measures whether that cost is per-call overhead or per-pair work, by
packing P endpoint pairs into ONE discovery. Channel's PropagationRequest has
no pairing restriction, so a packed call evaluates the full P x P cross
product; if the cost is still far below P separate calls, batching pays even
with that waste, and the only thing missing is a way to consume the result per
probe. The report next to this tool records that conclusion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from witwin.core.dynamics import DynamicScene

from witwin.radar.channel import ChannelPropagationAdapter, compile_scene
from witwin.radar.propagation import RadarEndpointSpec

FC = 77e9


def _endpoints(count, base, position, source):
    """``count`` distinct endpoints, each a 1 mm step from ``position`` [m].

    Distinct positions and distinct stable IDs, because a packed discovery is
    only meaningful if every row can be attributed to one probe afterwards.
    """

    offsets = torch.arange(count, dtype=torch.float32, device="cuda")[:, None] * 1e-3
    return RadarEndpointSpec(
        torch.arange(base, base + count, dtype=torch.int64, device="cuda"),
        torch.as_tensor(position, dtype=torch.float32, device="cuda")[None, :].repeat(count, 1) + offsets,
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda").repeat(count, 1),
        torch.ones(count, device="cuda") if source else None,
    )


def measure(counts):
    from validate_heavy_multipath import LinearPoint, room

    scene = room()
    compiled = compile_scene(DynamicScene(scene).at(0.0), reference_frequency_hz=FC)
    adapter = ChannelPropagationAdapter(
        compiled, reference_frequency_hz=FC, components=frozenset({"los", "reflection"}), max_depth=2
    )
    site = LinearPoint(torch.device("cuda")).at(0.0).positions_m[0].tolist()

    rows = []
    for count in counts:
        sources = _endpoints(count, 1_000_000, [0.0, 0.0, 0.0], True)
        sinks = _endpoints(count, 3_000_000, site, False)
        adapter.freeze(sources, sinks)  # warm the native route
        torch.cuda.synchronize()
        start = time.perf_counter()
        frozen = adapter.freeze(sources, sinks)
        torch.cuda.synchronize()
        rows.append(
            {
                "packed_probes": count,
                "discovered_pairs": count * count,
                "seconds": time.perf_counter() - start,
                "path_rows": int(frozen.row_count),
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    single = rows[0]["seconds"]
    for row in rows:
        row["separate_calls_seconds"] = single * row["packed_probes"]
        row["batching_speedup"] = row["separate_calls_seconds"] / row["seconds"]
    return {
        "fixed_overhead_seconds_estimate": single - (rows[-1]["seconds"] - single) / (rows[-1]["discovered_pairs"] - 1),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8, 19, 38])
    parser.add_argument("--output", type=Path, default=Path("output/discovery-batching"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = measure(args.counts)
    print(json.dumps({"fixed_overhead_seconds_estimate": result["fixed_overhead_seconds_estimate"]}), flush=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
