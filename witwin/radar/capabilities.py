"""What ``witwin.radar`` can do, as a versioned record.

Channel has had a capability manifest since Plan 07 and Radar has been reading
it (``channel.py``) without publishing one of its own.
That asymmetry is Phase-10 work item 3: a consumer that cannot describe itself
cannot be validated by its own consumers either.

Three rules this record obeys, each of which is the reason for a design choice
that would otherwise look arbitrary:

**It never triggers a Channel import.** ``propagation_consumer`` is embedded
only when ``witwin.channel.propagation.consumer`` is ALREADY in
``sys.modules``. Acceptance criterion A2 is the measured property that
importing ``witwin.radar`` loads zero ``witwin.channel`` modules; a capability
call that imported Channel to describe it would destroy exactly that property,
and would do it from the one function most likely to be called by a diagnostic
script.

**It restates nothing it does not own.** Where the Channel consumer contract is
present, its record is embedded verbatim rather than paraphrased. Where it is
absent, the field says so instead of falling back to a stale copy.

**The AD block is a summary of ``docs/dev/radar-ad-capability-matrix.md``, not
a second source.** The matrix is the authority, it is machine-parsed by
``tests/test_phase9_capability_matrix.py``, and it is a document rather than a
wheel member - so this record carries the summary a runtime caller needs and
``tests/test_phase10_diagnostics.py`` pins the summary against the matrix. Two
places, one direction of truth, and a test in between.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from typing import Any

CONSUMER_MODULE = "witwin.channel.propagation.consumer"


_CAPABILITIES: dict[str, Any] = {
    "schema_version": 1,
    "native_library": {
        "logical_owner": "_radar_native",
        "operator_families": [
            "fmcw_beat_synthesis",
            "fmcw_spectrum_synthesis",
            "ofdm_cfr_synthesis",
            "pulsed_echo_synthesis",
            "two_way_join",
            "scatter_response_aspect",
            "sensor_weight",
            "frontend_chain",
        ],
        "numerical_owner": "radar",
        "registry": "ci/native-binding-manifest.json",
    },
    "waveforms": {
        "fmcw_beat": {
            "axes": "chirp x segment x sample",
            "supports_tdm": True,
            "supports_wideband_band": True,
            "phasor": "conjugated_beat_domain",
        },
        "ofdm_cfr": {
            "axes": "symbol x segment x subcarrier",
            "supports_tdm": False,
            "supports_wideband_band": True,
            "phasor": "channel_domain",
        },
        "pulsed_echo": {
            "axes": "pulse x segment x sample",
            "supports_tdm": False,
            "supports_wideband_band": False,
            "phasor": "channel_domain",
            "pulse_kinds": ["rect", "lfm"],
        },
    },
    # The propagation request Radar actually makes. Narrower than what the
    # Channel consumer offers, and narrower on purpose: every component Radar
    # asks for must be freezable, because the per-frame route is a frozen
    # topology replay rather than a rediscovery.
    "propagation_request": {
        "response": "scalar_transport",
        "components": ["los", "reflection"],
        "topology_mode": "fixed_topology",
        "refused_components": ["diffraction", "transmission", "scattering"],
        "refusal_site": "witwin/radar/channel.py",
    },
    # A summary of docs/dev/radar-ad-capability-matrix.md. The four states are
    # that document's closed vocabulary; SILENT is not one of them and never
    # becomes one.
    "ad_contract": {
        "states": ["SUP", "ZERO", "REF", "DECL"],
        "modes": ["none", "jvp", "vjp"],
        "first_order_only": True,
        "higher_order_owner": "witwin/radar/policy.py",
        "production_finite_differences": False,
        "supported_leaves": [
            "mesh_vertices",
            "material_eps_r",
            "material_sigma_e",
            "site_positions",
            "transmitter_positions",
            "receiver_positions",
            "response_sigma_m2",
            "response_phase_rad",
        ],
        "refused_leaves": [
            "endpoint_powers_w",
            "endpoint_polarizations",
            "velocities_m_per_s",
            "smpl_pose",
            "smpl_shape",
            "waveform_spec_scalars",
            "frontend_spec_scalars",
            "sensor_pattern_tables",
            "sensor_frozen_geometry",
        ],
        "velocity_role": "forward_tangent_direction_never_a_leaf",
        "matrix_document": "docs/dev/radar-ad-capability-matrix.md",
    },
    # R-ADR-018's wall. Above it the chain is smooth and stays live; below the
    # first discrete decision every stage refuses at its entry, before any
    # device work and before any result object exists.
    "processing_wall": {
        "owner": "witwin/radar/policy.py",
        "differentiable_stages": [
            "range_profile",
            "range_doppler_map",
            "beam_cube",
            "matched_filter",
            "tdm_compensate",
            "music_spectrum",
            "music_image",
        ],
        "refusing_stages": [
            "adc_quantize",
            "ca_cfar",
            "ca_cfar_fast",
            "os_cfar",
            "ca_cfar_1d",
            "point_cloud",
            "peak_selection",
            "phase_comparison_aoa",
            "fft2_aoa",
            "tracking_handoff",
        ],
        "refuses_before_any_compute": True,
    },
    "host_observations": {
        "per_frame_owner": "none",
        "per_frame_budget": 2,
        "freeze_time_owner": "witwin/radar/paths.py",
        "contract": "docs/dev/standards/radar-adr-006-compact-contract-and-cardinality-budget.md",
    },
}


def _deployment_record() -> dict[str, Any]:
    from .deployment import DECLARED_SM_ARCHITECTURES, PTX_FORWARD_COMPATIBILITY_SM, VERIFIED_SM_ARCHITECTURES

    return {
        "declared_sm_architectures": list(DECLARED_SM_ARCHITECTURES),
        "verified_sm_architectures": list(VERIFIED_SM_ARCHITECTURES),
        "ptx_forward_compatibility_sm": PTX_FORWARD_COMPATIBILITY_SM,
    }


def _propagation_consumer_record() -> dict[str, Any]:
    """Channel's consumer contract, but only if it is already loaded.

    ``sys.modules`` rather than ``importlib.util.find_spec``: find_spec answers
    "could this be imported", which is true in every install that has Channel,
    and answering it would still require importing the module to read the
    record. The question this function asks is "is Channel already part of this
    process", and only ``sys.modules`` answers that.
    """

    module = sys.modules.get(CONSUMER_MODULE)
    if module is None:
        return {
            "status": "not_loaded",
            "note": (
                "witwin.channel is an optional dependency and reading this "
                "record must never load it; import "
                f"{CONSUMER_MODULE} first if the contract is wanted here"
            ),
        }
    record = module.capabilities()
    return {
        "status": "loaded",
        "contract_version": record.contract_version,
        "components": sorted(record.components),
        "responses": sorted(record.responses),
        "topology_modes": sorted(record.topology_modes),
        "ad_modes": sorted(record.ad_modes),
        "fixed_topology_components": sorted(record.fixed_topology_components),
        "supports_fixed_topology": record.supports_fixed_topology,
    }


def capabilities() -> dict[str, Any]:
    """Return the versioned Radar capability manifest."""

    from .cuda.runtime import RADAR_ABI_VERSION

    manifest = deepcopy(_CAPABILITIES)
    manifest["radar_abi_version"] = RADAR_ABI_VERSION
    manifest["deployment"] = _deployment_record()
    manifest["propagation_consumer"] = _propagation_consumer_record()
    return manifest


__all__ = ["capabilities"]
