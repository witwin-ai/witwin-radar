"""The wheel is either loadable or it is not built (Phase 10, work item 5).

Two owners are exercised here and neither of them is a comment:

* ``hatch_build.CustomBuildHook`` decides whether a wheel may be produced at
  all. Before Phase 10 it returned silently with no prebuilt and hatchling
  emitted a valid-looking ``py3-none-any`` wheel that no install could import.
* ``ci/wheel_smoke.py`` decides whether a produced wheel is a supported
  artifact. Its identity half is the interesting part: a wheel can ship a
  binary and a record of a DIFFERENT binary, and the only place that is cheap
  to notice is the packaging gate.

Every negative case below mutates a good input into a bad one and requires the
owner to reject it, because a check that has never been observed to fail is not
evidence that it can.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


RADAR_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, RADAR_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wheel_smoke():
    return _load("_phase10_wheel_smoke", "ci/wheel_smoke.py")


@pytest.fixture(scope="module")
def hatch_hook():
    pytest.importorskip("hatchling")
    return _load("_phase10_hatch_build", "hatch_build.py")


class _Hook:
    """The smallest thing ``CustomBuildHook.initialize`` actually reads."""

    def __init__(self, module, root: Path, target: str = "wheel"):
        self.hook = module.CustomBuildHook.__new__(module.CustomBuildHook)
        # ``root`` and ``target_name`` are read-only properties over
        # name-mangled attributes on hatchling's BuildHookInterface. Setting
        # the mangled names constructs the hook without running hatchling's
        # own constructor, which wants a full builder context this test has no
        # use for.
        self.hook.__dict__["_BuildHookInterface__root"] = str(root)
        self.hook.__dict__["_BuildHookInterface__target_name"] = target

    def initialize(self) -> dict:
        data: dict = {}
        self.hook.initialize("0.3.0", data)
        return data


def _prebuilt(root: Path, *, binary: bool = True, sidecars: bool = True) -> Path:
    directory = root / "witwin" / "radar" / "cuda" / "prebuilt"
    directory.mkdir(parents=True, exist_ok=True)
    if binary:
        (directory / "_radar_native.pyd").write_bytes(b"MZ not a real binary")
    if sidecars:
        (directory / "_radar_native.build-info.json").write_text("{}", encoding="utf-8")
        (directory / "_radar_native.build-fingerprint").write_text("0" * 64, encoding="utf-8")
    return directory


# ---------------------------------------------------------------------------
# The build hook
# ---------------------------------------------------------------------------


def test_a_complete_prebuilt_tags_the_wheel_for_this_platform(hatch_hook, tmp_path):
    _prebuilt(tmp_path)
    data = _Hook(hatch_hook, tmp_path).initialize()
    assert data["pure_python"] is False
    assert data["tag"].startswith("py3-none-")
    assert data["tag"] != "py3-none-any"


def test_a_missing_prebuilt_refuses_to_build_a_wheel(hatch_hook, tmp_path, monkeypatch):
    monkeypatch.delenv(hatch_hook.ALLOW_PURE_WHEEL_ENV, raising=False)
    (tmp_path / "witwin" / "radar" / "cuda" / "prebuilt").mkdir(parents=True)
    with pytest.raises(RuntimeError) as error:
        _Hook(hatch_hook, tmp_path).initialize()
    message = str(error.value)
    assert "no packaged radar extension" in message
    assert "build_radar_cuda_prebuilt.py" in message
    assert hatch_hook.ALLOW_PURE_WHEEL_ENV in message


def test_the_pure_wheel_opt_out_must_be_asked_for_explicitly(hatch_hook, tmp_path, monkeypatch):
    (tmp_path / "witwin" / "radar" / "cuda" / "prebuilt").mkdir(parents=True)
    monkeypatch.setenv(hatch_hook.ALLOW_PURE_WHEEL_ENV, "1")
    assert _Hook(hatch_hook, tmp_path).initialize() == {}


def test_a_binary_without_its_identity_sidecars_refuses_to_build(hatch_hook, tmp_path, monkeypatch):
    """The wheel would install and then fail at the user's first import.

    R-ADR-019 validates the binary against both sidecars before
    ``torch.ops.load_library``, so this is not a strictness preference: a wheel
    packed without them cannot load at all.
    """

    monkeypatch.delenv(hatch_hook.ALLOW_PURE_WHEEL_ENV, raising=False)
    _prebuilt(tmp_path, sidecars=False)
    with pytest.raises(RuntimeError, match="has no build identity"):
        _Hook(hatch_hook, tmp_path).initialize()


def test_two_binaries_refuse_to_build(hatch_hook, tmp_path, monkeypatch):
    monkeypatch.delenv(hatch_hook.ALLOW_PURE_WHEEL_ENV, raising=False)
    directory = _prebuilt(tmp_path)
    (directory / "_radar_native.so").write_bytes(b"\x7fELF not a real binary")
    with pytest.raises(RuntimeError, match="multiple packaged radar extensions"):
        _Hook(hatch_hook, tmp_path).initialize()


def test_an_sdist_build_is_not_the_hooks_business(hatch_hook, tmp_path, monkeypatch):
    monkeypatch.delenv(hatch_hook.ALLOW_PURE_WHEEL_ENV, raising=False)
    assert _Hook(hatch_hook, tmp_path, target="sdist").initialize() == {}


# ---------------------------------------------------------------------------
# The packaging declarations the loader depends on
# ---------------------------------------------------------------------------


def test_the_wheel_declares_the_two_identity_sidecars_as_artifacts():
    text = (RADAR_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for pattern in (
        '"witwin/radar/cuda/prebuilt/*.pyd"',
        '"witwin/radar/cuda/prebuilt/*.so"',
        '"witwin/radar/cuda/prebuilt/*.build-info.json"',
        '"witwin/radar/cuda/prebuilt/*.build-fingerprint"',
    ):
        assert pattern in text, pattern


def test_channel_is_an_optional_extra_and_never_a_required_dependency():
    """R-ADR-008's decision, as a check rather than a paragraph."""

    import tomllib

    data = tomllib.loads((RADAR_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    required = data["project"]["dependencies"]
    assert not any("channel" in entry for entry in required), required
    extras = data["project"]["optional-dependencies"]
    assert extras["channel"] == ["witwin-channel>=0.4,<0.5"]
    for entry in required + [item for value in extras.values() for item in value]:
        assert "rayd" not in entry.lower(), entry


# ---------------------------------------------------------------------------
# The wheel smoke's identity chain
# ---------------------------------------------------------------------------


def _sources_digest(payloads: dict[str, bytes], order) -> str:
    digest = hashlib.sha256()
    for member in order:
        digest.update(Path(member).name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payloads[member])
        digest.update(b"\0")
    return digest.hexdigest()


def _make_wheel(wheel_smoke, path: Path, *, mutate=None) -> Path:
    """A minimal archive carrying only what ``_audit_identity`` reads."""

    binary = b"MZ" + b"\x00" * 512
    sources = {member: f"// {member}\n".encode("utf-8") for member in wheel_smoke._SOURCE_MEMBERS}
    record = {
        "binary_sha256": hashlib.sha256(binary).hexdigest(),
        "build_type": "developer",
        "compiler": "msvc 19.44.35207.1",
        "cuda_architectures": ["120"],
        "cuda_compiler_version": "12.9.41",
        "cuda_version": "12.8",
        "cxx_abi": "msvc",
        "extension_name": "_radar_native",
        "operator_symbols": ["forward_chunked"],
        "platform_tag": "win_amd64",
        "python_abi": "stable-abi-v1",
        "radar_abi_version": 1,
        "radar_git_dirty": False,
        "radar_git_sha": "0" * 40,
        "source_fingerprint": _sources_digest(sources, wheel_smoke._SOURCE_MEMBERS),
        "torch_target_version": "0x020a000000000000",
        "torch_version": "2.10.0",
    }
    record["build_fingerprint"] = hashlib.sha256(b"fingerprint").hexdigest()
    members = {
        "witwin/radar/cuda/prebuilt/_radar_native.pyd": binary,
        "witwin/radar/cuda/prebuilt/_radar_native.build-info.json": None,
        "witwin/radar/cuda/prebuilt/_radar_native.build-fingerprint": None,
        **sources,
    }
    if mutate is not None:
        mutate(record, members)
    members["witwin/radar/cuda/prebuilt/_radar_native.build-info.json"] = json.dumps(
        record
    ).encode("utf-8")
    if members["witwin/radar/cuda/prebuilt/_radar_native.build-fingerprint"] is None:
        members["witwin/radar/cuda/prebuilt/_radar_native.build-fingerprint"] = (
            record["build_fingerprint"].encode("ascii") + b"\n"
        )
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            if payload is not None:
                archive.writestr(name, payload)
    return path


def _audit(wheel_smoke, path: Path):
    with zipfile.ZipFile(path) as archive:
        return wheel_smoke._audit_identity(
            archive, "witwin/radar/cuda/prebuilt/_radar_native.pyd"
        )


def test_a_consistent_wheel_passes_the_identity_audit(wheel_smoke, tmp_path):
    record = _audit(wheel_smoke, _make_wheel(wheel_smoke, tmp_path / "good.whl"))
    assert record["extension_name"] == "_radar_native"


def test_a_swapped_binary_is_caught_by_the_recorded_digest(wheel_smoke, tmp_path):
    def mutate(record, members):
        members["witwin/radar/cuda/prebuilt/_radar_native.pyd"] = b"MZ" + b"\x01" * 512

    wheel = _make_wheel(wheel_smoke, tmp_path / "swapped.whl", mutate=mutate)
    with pytest.raises(wheel_smoke.WheelSmokeError, match="binary digest"):
        _audit(wheel_smoke, wheel)


def test_repacked_sources_are_caught_by_the_source_fingerprint(wheel_smoke, tmp_path):
    """The wheel packed sources it did not build from.

    This is the check the loader repeats at import time, so a wheel that fails
    it is a wheel that cannot load - which makes catching it here strictly
    earlier, not merely stricter.
    """

    def mutate(record, members):
        members["witwin/radar/cuda/kernels/fmcw_beat.cu"] = b"// a different revision\n"

    wheel = _make_wheel(wheel_smoke, tmp_path / "repacked.whl", mutate=mutate)
    with pytest.raises(wheel_smoke.WheelSmokeError, match="source_fingerprint"):
        _audit(wheel_smoke, wheel)


def test_a_fingerprint_sidecar_that_disagrees_is_refused(wheel_smoke, tmp_path):
    def mutate(record, members):
        members["witwin/radar/cuda/prebuilt/_radar_native.build-fingerprint"] = (
            b"f" * 64 + b"\n"
        )

    wheel = _make_wheel(wheel_smoke, tmp_path / "fingerprint.whl", mutate=mutate)
    with pytest.raises(wheel_smoke.WheelSmokeError, match="build_fingerprint"):
        _audit(wheel_smoke, wheel)


def test_a_missing_sidecar_is_refused(wheel_smoke, tmp_path):
    wheel = tmp_path / "nosidecar.whl"
    _make_wheel(wheel_smoke, wheel)
    stripped = tmp_path / "stripped.whl"
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(stripped, "w") as target:
        for name in source.namelist():
            if name.endswith(".build-info.json"):
                continue
            target.writestr(name, source.read(name))
    with pytest.raises(wheel_smoke.WheelSmokeError, match="identity sidecar"):
        _audit(wheel_smoke, stripped)


def test_a_record_naming_another_extension_is_refused(wheel_smoke, tmp_path):
    def mutate(record, members):
        record["extension_name"] = "witwin_radar_dirichlet_cuda"

    wheel = _make_wheel(wheel_smoke, tmp_path / "othername.whl", mutate=mutate)
    with pytest.raises(wheel_smoke.WheelSmokeError, match="expected '_radar_native'"):
        _audit(wheel_smoke, wheel)


def test_the_native_member_is_discovered_by_suffix_not_by_name(wheel_smoke):
    """A second, differently named DSO must fail rather than be ignored."""

    good = ["witwin/radar/__init__.py", "witwin/radar/cuda/prebuilt/_radar_native.pyd"]
    assert wheel_smoke._native_member(good) == good[1]
    with pytest.raises(wheel_smoke.WheelSmokeError, match="exactly one native member"):
        wheel_smoke._native_member(good + ["witwin/radar/vendored_runtime.dll"])
    with pytest.raises(wheel_smoke.WheelSmokeError, match="native member must be"):
        wheel_smoke._native_member(
            ["witwin/radar/cuda/prebuilt/witwin_radar_dirichlet_cuda.pyd"]
        )
