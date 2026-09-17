"""Completed-GPU public-scene benchmark against saved cubes and an ADC oracle.

Uses the unchanged MATLAB-comparison scene definitions. Baseline inputs must
come from an independently run pre-optimization checkout, never this script.
Latency is evidence, not a portable CI threshold on a shared desktop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compare_matlab_scenarios import independent_free_space_oracle, l2_norm, motion_export, timing_summary
from scipy.io import loadmat


def validate(output, baseline, cases):
    motion_export(output, cases, accuracy_only=False)
    results = {}
    for case in cases:
        path = output / f"{case}-motion-input.mat"
        source = loadmat(path, squeeze_me=True)
        oracle = independent_free_space_oracle(source, case)
        iq_error = float(l2_norm(source["witwin_iq"] - oracle) / l2_norm(oracle))
        entry = {
            "timing": timing_summary(source["seconds"]),
            "oracle_iq_relative_l2": iq_error,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "diagnostics": json.loads((output / f"{case}-diagnostics.json").read_text()),
        }
        old_path = baseline / f"{case}-motion-input.mat"
        if old_path.exists():
            previous = loadmat(old_path, squeeze_me=True)
            for field in ("fs", "fc", "samples", "chirps", "period", "slope"):
                assert source[field] == previous[field], (case, field)
            change = float(l2_norm(source["witwin_iq"] - previous["witwin_iq"]) / l2_norm(previous["witwin_iq"]))
            entry.update(
                baseline=timing_summary(previous["seconds"]),
                baseline_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                iq_change_relative_l2=change,
            )
            entry["speedup"] = entry["baseline"]["median_seconds"] / entry["timing"]["median_seconds"]
            assert change < 2e-5, entry
        results[case] = entry
        assert iq_error < 0.012, entry
    (output / "acceptance.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=("rotor", "acceleration", "limbs", "static"), default=["rotor"])
    args = parser.parse_args()
    validate(args.output, args.baseline, args.cases)
