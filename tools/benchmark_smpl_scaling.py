"""Round-robin site-count scaling for the same live SMPL walking workload.

Run separately from the long-sequence benchmark, never concurrently. Each
round interleaves the site counts at the same body instant so desktop load
drift is less likely to masquerade as a count-dependent cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_smpl_walk import WalkingBody, completed, digest, make_session, simulate, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--sites", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("output/smpl-walk/scaling"))
    args = parser.parse_args()
    if args.rounds < 1 or min(args.sites) < 1:
        parser.error("rounds and site counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {}
    for count in args.sites:
        body = WalkingBody(args.motion, args.model_root, count, 10)
        cases[count] = make_session(body)
        completed(lambda count=count: simulate(*cases[count], 2.3))
    rows = []
    for repeat in range(args.rounds):
        for instant in (0.0, 2.3):
            for count in args.sites:
                result, elapsed = completed(lambda count=count, instant=instant: simulate(*cases[count], instant))
                row = {
                    "round": repeat,
                    "time_s": instant,
                    "sites": count,
                    "ms": elapsed * 1000,
                    "diagnostics": dict(result.adaptive_diagnostics[0]),
                }
                rows.append(row)
                with (args.output / "measurements.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps({key: value for key, value in row.items() if key != "diagnostics"}), flush=True)
                del result
    record = {
        "motion_sha256": digest(args.motion),
        "script_sha256": digest(__file__),
        "workload_script_sha256": digest(Path(__file__).with_name("benchmark_smpl_walk.py")),
        "measurements": rows,
        "summary": {str(count): summary([row["ms"] for row in rows if row["sites"] == count]) for count in args.sites},
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
