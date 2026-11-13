"""What this build of witwin-radar supports, as one versioned record.

:func:`capabilities` is Radar's counterpart of ``witwin.channel.capabilities()``:
plain JSON data, a fresh copy per call, ``schema_version`` first. Nothing in it
is written here. Every entry is read from the module that owns the
vocabulary - the waveform records, the synthesis specs, :class:`Motion`, the
round-trip composer, the processing package and the native build record - so
a vocabulary that grows in its owner grows here without an edit, and this
record cannot claim what the code refuses.

Importing this module does not load the native library or Channel. The native
operator list is read from the prebuilt's build record on disk, and Channel's
consumer contract is reported only when Channel is installed.
"""

from __future__ import annotations

from typing import Any

#: Bumped when a key is removed or changes meaning; adding a key keeps it.
SCHEMA_VERSION = 1


def _leaves(owner: type) -> dict[str, list[str]]:
    """``field -> AD modes`` an owner declares in its ``LEAVES`` mapping."""

    return {name: sorted(modes) for name, modes in owner.LEAVES.items()}  # type: ignore[attr-defined]


def _native() -> dict[str, Any]:
    """The prebuilt's identity from its build record, without loading it."""

    from .cuda import runtime

    binary = runtime.prebuilt_extension_path()
    record: dict[str, Any] = {"abi_version": runtime.RADAR_ABI_VERSION, "available": False, "operators": []}
    try:
        info = runtime.read_build_info(binary)
    except (runtime.RadarExtensionLoadError, runtime.RadarExtensionABIError) as error:
        record["reason"] = str(error)
        return record
    operators = info["operator_symbols"]
    architectures = info["cuda_architectures"]
    # read_build_info has already type-checked both as lists.
    assert isinstance(operators, list) and isinstance(architectures, list)
    record.update(
        available=True,
        operators=sorted(str(name) for name in operators),
        cuda_architectures=[str(arch) for arch in architectures],
        torch_version=str(info["torch_version"]),
    )
    return record


def _channel() -> dict[str, Any] | None:
    """Channel's consumer contract as Radar reaches it, or ``None`` without Channel."""

    try:
        from .channel import consumer_capabilities
    except ImportError:
        return None
    return consumer_capabilities()


def capabilities() -> dict[str, Any]:
    """Return the versioned capability record of this build (a fresh copy each call).

    Keys: ``waveforms`` (per kind, the output domains, receiver sampling,
    MIMO schemes, modulations and pulse kinds the records accept),
    ``motion`` (sampling kinds, discovery cadences, Channel world motions),
    ``propagation`` (the Channel components the verbs request and the
    round-trip row classes Radar composes), ``targets`` (the target records),
    ``ad_leaves`` (per public record, each differentiable field and the AD
    modes it supports; the authority for every cell is
    ``docs/dev/radar-ad-capability-matrix.md``), ``processing`` (windows,
    detectors, angle routes, clutter filters, point-cloud and image frames),
    ``native`` (ABI version and the operators of the installed prebuilt) and
    ``channel`` (Channel's consumer contract, or ``None``).
    """

    from .deployment import package_version
    from .facets import MATERIAL_LEAVES
    from .motion import CADENCES, MOTION_KINDS, WORLD_MOTIONS
    from .paths import COMPONENT_NAMES
    from .processing import AOA_ROUTES, CFAR_KINDS, CLUTTER_FILTERS, IMAGE_FRAMES, POINT_CLOUD_FRAMES
    from .processing.signal import WINDOWS
    from .radar import FMCW_SAMPLINGS, PROPAGATION_COMPONENTS, WAVEFORM_KINDS, Radar
    from .sensors import PATTERN_KINDS, PATTERN_QUANTITIES, Coupling, Pattern
    from .smpl import SMPLBody
    from .targets import PointTargets
    from .targets import __all__ as target_records
    from .synthesis.assembly import (
        FMCW_OUTPUT_DOMAINS,
        MIMO_SCHEMES,
        OFDM_MODULATIONS,
        OFDM_OUTPUT_DOMAINS,
        PULSE_KINDS,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": package_version(),
        "waveforms": {
            "kinds": list(WAVEFORM_KINDS),
            "fmcw": {
                "outputs": list(FMCW_OUTPUT_DOMAINS),
                "sampling": list(FMCW_SAMPLINGS),
                "mimo": list(MIMO_SCHEMES),
            },
            "ofdm": {"outputs": list(OFDM_OUTPUT_DOMAINS), "modulations": list(OFDM_MODULATIONS)},
            "pulsed": {"pulse_kinds": sorted(PULSE_KINDS)},
        },
        "motion": {"kinds": list(MOTION_KINDS), "cadences": list(CADENCES), "world_motions": list(WORLD_MOTIONS)},
        "propagation": {"components": list(PROPAGATION_COMPONENTS), "round_trip_classes": list(COMPONENT_NAMES)},
        "targets": sorted(target_records),
        "patterns": {"kinds": list(PATTERN_KINDS), "quantities": list(PATTERN_QUANTITIES)},
        "ad_leaves": {
            "Radar": _leaves(Radar),
            "PointTargets": _leaves(PointTargets),
            "Pattern": _leaves(Pattern),
            "Coupling": _leaves(Coupling),
            "SMPLBody": _leaves(SMPLBody),
            "material": {name: sorted(modes) for name, modes in MATERIAL_LEAVES.items()},
        },
        "processing": {
            "windows": list(WINDOWS),
            "cfar": list(CFAR_KINDS),
            "aoa_routes": sorted(AOA_ROUTES),
            "clutter_filters": list(CLUTTER_FILTERS),
            "point_cloud_frames": list(POINT_CLOUD_FRAMES),
            "image_frames": list(IMAGE_FRAMES),
        },
        "native": _native(),
        "channel": _channel(),
    }


__all__ = ["SCHEMA_VERSION", "capabilities"]
