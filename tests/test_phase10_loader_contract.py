"""The radar native library loads from one place, or fails saying why.

R-ADR-019. Every case here runs in a FRESH subprocess with an explicit
environment, because the property under test is process-global: once
``torch.ops.load_library`` has run, the library cannot be unloaded, and once
``torch.utils.cpp_extension`` has prepared the MSVC environment the damage to
``os.environ`` is permanent for that process. A monkeypatched in-process test
would prove nothing about either.

Each case asserts on the exception TYPE and on the part of the message that
tells an operator what to do, not merely that something raised.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from witwin.radar.cuda import runtime as build
identity = build


REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = REPO_ROOT / "witwin" / "radar" / "cuda"

#: Loads ``runtime.py`` from an arbitrary directory as a standalone package, so a
#: case can point the loader at a copied source tree without touching the
#: checkout. ``sys.argv[1]`` is that directory.
_DRIVER = r"""
import importlib.util, json, os, sys, types

cuda_dir = sys.argv[1]
package = types.ModuleType("radar_cuda_probe")
package.__path__ = [cuda_dir]
sys.modules["radar_cuda_probe"] = package


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package.runtime = _load("radar_cuda_probe.runtime", os.path.join(cuda_dir, "runtime.py"))
build = package.runtime

result = {}
try:
    module = build.build_extension()
    info = module.build_info()
    native = info["native_build"]
    result = {
        "ok": True,
        "origin": info["origin"],
        "extension_path": info["extension_path"],
        "radar_abi_version": info["radar_abi_version"],
        "symbols": len(native["operator_symbols"]) if native else 0,
        "build_type": native["build_type"] if native else None,
    }
except BaseException as exc:  # noqa: BLE001 - the exception IS the assertion
    result = {"ok": False, "type": type(exc).__name__, "message": str(exc)}

result["cpp_extension_imported"] = "torch.utils.cpp_extension" in sys.modules
print("PHASE10RESULT " + json.dumps(result))
"""


def _run_driver(cuda_dir: Path, env: dict[str, str]) -> dict:
    child_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("WITWIN_RADAR_")
    }
    child_env.update(env)
    completed = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(cuda_dir)],
        capture_output=True,
        text=True,
        env=child_env,
        cwd=str(REPO_ROOT),
        check=False,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("PHASE10RESULT "):
            return json.loads(line[len("PHASE10RESULT ") :])
    raise AssertionError(
        f"driver produced no result\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _copy_cuda_tree(destination: Path) -> Path:
    """A standalone copy of the loader plus its sources and prebuilt artifact.

    Copied rather than monkeypatched because ``source_root()`` is
    ``Path(__file__).parent``: relocating the module relocates everything the
    contract depends on - the sources it fingerprints and the prebuilt directory
    it searches - in one move, and a case can then mutate either side.
    """

    cuda_dir = destination / "cuda"
    shutil.copytree(
        CUDA_DIR,
        cuda_dir,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return cuda_dir


def _prebuilt_in(cuda_dir: Path) -> Path:
    return cuda_dir / "prebuilt" / build.prebuilt_extension_path().name


def _rewrite_record(binary: Path, **changes: object) -> None:
    """Edit the record and RE-SIGN it, so the case tests what it means to.

    Editing ``torch_version`` alone would trip the fingerprint self-check first
    and never reach the runtime comparison, which would make the test pass for
    the wrong reason.
    """

    record_path = identity.build_info_sidecar_path(binary)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.update(changes)
    record["build_fingerprint"] = identity.compute_build_fingerprint(record)
    identity.write_sidecars(binary, record)


@pytest.fixture(scope="module")
def packaged_binary() -> Path:
    binary = build.prebuilt_extension_path()
    if not binary.exists():
        pytest.skip(
            "no packaged radar prebuilt; run scripts/build_radar_cuda_prebuilt.py"
        )
    return binary


def test_the_packaged_prebuilt_is_the_normal_load_source(packaged_binary):
    """And it validates the WHOLE recorded symbol set, not a family sample."""

    result = _run_driver(CUDA_DIR, {})
    assert result["ok"], result
    assert result["origin"] == "packaged"
    assert Path(result["extension_path"]) == packaged_binary
    assert result["radar_abi_version"] == identity.RADAR_ABI_VERSION
    manifest = json.loads(
        (REPO_ROOT / "ci" / "native-binding-manifest.json").read_text(encoding="utf-8")
    )
    assert result["symbols"] == len({entry["symbol"] for entry in manifest["operators"]})
    assert result["cpp_extension_imported"] is False


def test_a_missing_prebuilt_raises_and_does_not_compile(tmp_path, packaged_binary):
    """The defect this whole contract exists to remove.

    A missing prebuilt used to fall through to
    ``torch.utils.cpp_extension.load`` inside whatever process happened to
    import the package. On Windows that mutates ``PATH`` from ``vcvars64`` and
    the freshly built library then fails ``DllMain`` in the same process.
    """

    cuda_dir = _copy_cuda_tree(tmp_path)
    _prebuilt_in(cuda_dir).unlink()
    scratch_temp = tmp_path / "temp"
    scratch_temp.mkdir()

    result = _run_driver(
        cuda_dir,
        {"TMP": str(scratch_temp), "TEMP": str(scratch_temp), "TMPDIR": str(scratch_temp)},
    )
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionLoadError", result
    for variable in (
        "WITWIN_RADAR_NATIVE_DEVELOPER_OVERRIDE",
        "WITWIN_RADAR_NATIVE_EXTENSION_PATH",
        "WITWIN_RADAR_NATIVE_EXPECTED_FINGERPRINT",
    ):
        assert variable in result["message"], result["message"]
    assert result["cpp_extension_imported"] is False
    assert not list(scratch_temp.glob(f"{build.EXTENSION_NAME}*")), sorted(
        path.name for path in scratch_temp.iterdir()
    )


def test_a_binary_without_its_record_is_refused(tmp_path, packaged_binary):
    cuda_dir = _copy_cuda_tree(tmp_path)
    identity.build_info_sidecar_path(_prebuilt_in(cuda_dir)).unlink()

    result = _run_driver(cuda_dir, {})
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionLoadError", result
    assert "build record" in result["message"]
    assert "build_radar_cuda_prebuilt" in result["message"]


def test_a_binary_without_its_fingerprint_sidecar_is_refused(tmp_path, packaged_binary):
    cuda_dir = _copy_cuda_tree(tmp_path)
    identity.fingerprint_sidecar_path(_prebuilt_in(cuda_dir)).unlink()

    result = _run_driver(cuda_dir, {})
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionLoadError", result
    assert "build fingerprint" in result["message"]


def test_a_corrupted_fingerprint_sidecar_is_refused(tmp_path, packaged_binary):
    cuda_dir = _copy_cuda_tree(tmp_path)
    sidecar = identity.fingerprint_sidecar_path(_prebuilt_in(cuda_dir))
    digest = sidecar.read_text(encoding="ascii").strip()
    flipped = ("0" if digest[0] != "0" else "1") + digest[1:]
    sidecar.write_text(flipped + "\n", encoding="ascii")

    result = _run_driver(cuda_dir, {})
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionABIError", result
    assert "fingerprint" in result["message"]


def test_a_byte_flipped_binary_is_refused(tmp_path, packaged_binary):
    """The check a compiled-in self-report cannot make.

    A ``build_info`` ABI symbol is regenerated by the same rebuild that produces
    a swap, so it agrees with whatever binary is present. A recorded digest of
    the bytes does not.
    """

    cuda_dir = _copy_cuda_tree(tmp_path)
    binary = _prebuilt_in(cuda_dir)
    payload = bytearray(binary.read_bytes())
    payload[-1] ^= 0xFF
    binary.write_bytes(payload)

    result = _run_driver(cuda_dir, {})
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionABIError", result
    assert "binary_sha256" in result["message"]


def test_a_mutated_source_tree_is_refused(tmp_path, packaged_binary):
    """Two revisions of the same operator set are no longer indistinguishable.

    This is the hole the family-name presence check left open: a binary built
    from other sources registers the same names and loads clean. It is also why
    the wheel must keep shipping the CUDA sources - they are part of the
    identity, not a leftover.
    """

    cuda_dir = _copy_cuda_tree(tmp_path)
    kernel = cuda_dir / "fmcw_beat.cu"
    kernel.write_bytes(kernel.read_bytes() + b"\n// phase10 mutation\n")

    result = _run_driver(cuda_dir, {})
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionABIError", result
    assert "source_fingerprint" in result["message"]


def test_a_record_built_for_another_torch_is_refused(tmp_path, packaged_binary):
    cuda_dir = _copy_cuda_tree(tmp_path)
    _rewrite_record(_prebuilt_in(cuda_dir), torch_version="1.13.0")

    result = _run_driver(cuda_dir, {})
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionABIError", result
    assert "torch_version" in result["message"]
    assert "1.13.0" in result["message"]


def test_a_record_from_another_abi_version_is_refused(tmp_path, packaged_binary):
    cuda_dir = _copy_cuda_tree(tmp_path)
    _rewrite_record(
        _prebuilt_in(cuda_dir),
        radar_abi_version=identity.RADAR_ABI_VERSION + 1,
    )

    result = _run_driver(cuda_dir, {})
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionABIError", result
    assert "ABI mismatch" in result["message"]


def test_a_record_with_an_unknown_field_is_refused(tmp_path, packaged_binary):
    """A record this loader does not fully understand is not a record."""

    cuda_dir = _copy_cuda_tree(tmp_path)
    record_path = identity.build_info_sidecar_path(_prebuilt_in(cuda_dir))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["invented_field"] = "surprise"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    result = _run_driver(cuda_dir, {})
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionABIError", result
    assert "invented_field" in result["message"]


@pytest.mark.parametrize(
    "override_env",
    [
        {"WITWIN_RADAR_NATIVE_DEVELOPER_OVERRIDE": "1"},
        {"WITWIN_RADAR_NATIVE_EXTENSION_PATH": "C:/nowhere/x.pyd"},
        {"WITWIN_RADAR_NATIVE_EXPECTED_FINGERPRINT": "0" * 64},
        {
            "WITWIN_RADAR_NATIVE_DEVELOPER_OVERRIDE": "1",
            "WITWIN_RADAR_NATIVE_EXTENSION_PATH": "C:/nowhere/x.pyd",
        },
    ],
)
def test_a_partial_developer_override_is_refused(override_env, packaged_binary):
    """Even though the packaged prebuilt is right there and would have loaded.

    Ignoring two of three variables because the default happened to work is how
    a developer ends up measuring a binary they did not select.
    """

    result = _run_driver(CUDA_DIR, override_env)
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionLoadError", result
    for variable in (
        "WITWIN_RADAR_NATIVE_DEVELOPER_OVERRIDE",
        "WITWIN_RADAR_NATIVE_EXTENSION_PATH",
        "WITWIN_RADAR_NATIVE_EXPECTED_FINGERPRINT",
    ):
        assert variable in result["message"], result["message"]


def test_a_complete_developer_override_loads(tmp_path, packaged_binary):
    cuda_dir = _copy_cuda_tree(tmp_path)
    developer_binary = _prebuilt_in(cuda_dir)
    packaged_copy = tmp_path / "developer"
    packaged_copy.mkdir()
    moved = packaged_copy / developer_binary.name
    for source in (
        developer_binary,
        identity.build_info_sidecar_path(developer_binary),
        identity.fingerprint_sidecar_path(developer_binary),
    ):
        shutil.move(str(source), str(packaged_copy / source.name))
    fingerprint = (
        identity.fingerprint_sidecar_path(moved).read_text(encoding="ascii").strip()
    )

    result = _run_driver(
        cuda_dir,
        {
            "WITWIN_RADAR_NATIVE_DEVELOPER_OVERRIDE": "1",
            "WITWIN_RADAR_NATIVE_EXTENSION_PATH": str(moved),
            "WITWIN_RADAR_NATIVE_EXPECTED_FINGERPRINT": fingerprint,
        },
    )
    assert result["ok"], result
    assert result["origin"] == "developer"
    assert Path(result["extension_path"]) == moved
    assert result["cpp_extension_imported"] is False


def test_a_developer_override_with_the_wrong_fingerprint_is_refused(
    tmp_path, packaged_binary
):
    cuda_dir = _copy_cuda_tree(tmp_path)
    developer_binary = _prebuilt_in(cuda_dir)
    packaged_copy = tmp_path / "developer"
    packaged_copy.mkdir()
    moved = packaged_copy / developer_binary.name
    for source in (
        developer_binary,
        identity.build_info_sidecar_path(developer_binary),
        identity.fingerprint_sidecar_path(developer_binary),
    ):
        shutil.move(str(source), str(packaged_copy / source.name))

    result = _run_driver(
        cuda_dir,
        {
            "WITWIN_RADAR_NATIVE_DEVELOPER_OVERRIDE": "1",
            "WITWIN_RADAR_NATIVE_EXTENSION_PATH": str(moved),
            "WITWIN_RADAR_NATIVE_EXPECTED_FINGERPRINT": "a" * 64,
        },
    )
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionABIError", result
    assert "expected fingerprint" in result["message"]


def test_a_relative_developer_override_path_is_refused(packaged_binary):
    result = _run_driver(
        CUDA_DIR,
        {
            "WITWIN_RADAR_NATIVE_DEVELOPER_OVERRIDE": "1",
            "WITWIN_RADAR_NATIVE_EXTENSION_PATH": "prebuilt/x.pyd",
            "WITWIN_RADAR_NATIVE_EXPECTED_FINGERPRINT": "b" * 64,
        },
    )
    assert result["ok"] is False, result
    assert result["type"] == "RadarExtensionLoadError", result
    assert "absolute path" in result["message"]


_JIT_ROUTE_DRIVER = r"""
import importlib.util, json, os, sys, types

cuda_dir = sys.argv[1]
package = types.ModuleType("radar_cuda_probe")
package.__path__ = [cuda_dir]
sys.modules["radar_cuda_probe"] = package


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package.runtime = _load("radar_cuda_probe.runtime", os.path.join(cuda_dir, "runtime.py"))
build = package.runtime

result = {"import_time_cpp_extension": "torch.utils.cpp_extension" in sys.modules}

# Route, not compile: the compiler itself is exercised by
# scripts/build_radar_cuda_prebuilt.py, which must never run in a test process.
reached = []
build._jit_build_extension = lambda **kwargs: reached.append(kwargs) or "JIT"
try:
    returned = build.build_extension()
    result["returned"] = returned if isinstance(returned, str) else "module"
except BaseException as exc:  # noqa: BLE001
    result["returned"] = None
    result["type"] = type(exc).__name__
    result["message"] = str(exc)
result["reached_jit"] = bool(reached)
print("PHASE10RESULT " + json.dumps(result))
"""


def test_the_jit_route_is_reachable_only_when_the_build_script_asks(
    tmp_path, packaged_binary
):
    """``WITWIN_RADAR_NATIVE_BUILD=1`` means "compile from these sources".

    It therefore bypasses the packaged artifact: a stale prebuilt that the
    loader refuses must not be able to block the one command that replaces it.
    Nothing else reaches the compiler.
    """

    cuda_dir = _copy_cuda_tree(tmp_path)

    def _probe(env: dict[str, str]) -> dict:
        child_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("WITWIN_RADAR_")
        }
        child_env.update(env)
        completed = subprocess.run(
            [sys.executable, "-c", _JIT_ROUTE_DRIVER, str(cuda_dir)],
            capture_output=True,
            text=True,
            env=child_env,
            cwd=str(REPO_ROOT),
            check=False,
        )
        for line in completed.stdout.splitlines():
            if line.startswith("PHASE10RESULT "):
                return json.loads(line[len("PHASE10RESULT ") :])
        raise AssertionError(completed.stdout + completed.stderr)

    asked = _probe({"WITWIN_RADAR_NATIVE_BUILD": "1"})
    assert asked["reached_jit"] is True, asked
    assert asked["returned"] == "JIT", asked
    # Lazy: importing the loader must not drag the compiler machinery in.
    assert asked["import_time_cpp_extension"] is False, asked

    not_asked = _probe({})
    assert not_asked["reached_jit"] is False, not_asked


def test_preparing_build_tools_without_asking_for_a_build_raises():
    """Checked in-process because it raises BEFORE it mutates anything.

    This is the second guard, at the point of damage: the function replaces the
    whole process ``PATH`` from ``vcvars64``, and a caller that reaches it
    without asking for a build has made a mistake that must not be absorbed.
    """

    for name in ("WITWIN_RADAR_NATIVE_BUILD",):
        assert os.environ.get(name) != "1", name
    with pytest.raises(build.RadarExtensionLoadError, match="WITWIN_RADAR_NATIVE_BUILD"):
        build._ensure_windows_build_tools_on_path()
    with pytest.raises(build.RadarExtensionLoadError, match="WITWIN_RADAR_NATIVE_BUILD"):
        build._ensure_cuda_home_from_nvcc()


def test_the_loader_does_not_import_the_compiler_machinery_at_module_scope():
    """A static companion to the subprocess evidence above.

    ``sys.modules`` proves today's behaviour; this proves the code shape that
    causes it, so a future edit that hoists the import back to the top fails
    here with a readable reason rather than in an unrelated CUDA test.
    """

    import ast

    tree = ast.parse((CUDA_DIR / "runtime.py").read_text(encoding="utf-8"))
    module_scope_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_scope_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_scope_imports.add(node.module)
    assert not any(
        "cpp_extension" in name for name in module_scope_imports
    ), sorted(module_scope_imports)

    jit = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_jit_build_extension"
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "torch.utils.cpp_extension"
        for node in ast.walk(jit)
    )


def test_the_recorded_symbol_set_matches_the_binding_manifest(packaged_binary):
    """Manifest, binary and loader agree, or the artifact is not publishable."""

    record = identity.read_build_info(packaged_binary)
    manifest = json.loads(
        (REPO_ROOT / "ci" / "native-binding-manifest.json").read_text(encoding="utf-8")
    )
    assert set(record["operator_symbols"]) == {
        entry["symbol"] for entry in manifest["operators"]
    }


def test_the_packaged_record_validates_against_this_checkout(packaged_binary):
    info = identity.validate_identity(packaged_binary, build.extension_sources())
    assert info["extension_name"] == build.EXTENSION_NAME
    assert info["radar_abi_version"] == identity.RADAR_ABI_VERSION
    assert info["build_type"] in identity.BUILD_TYPES
    assert info["torch_target_version"] == build.TORCH_TARGET_VERSION
    assert info["source_fingerprint"].startswith(build.source_fingerprint())
