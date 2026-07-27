"""Install Core, Channel and Radar together and measure that they coexist.

``ci/wheel_smoke.py`` audits ONE wheel and proves it imports from a fresh
install. This script asks the question that only the consumer can ask, and that
is why it lives in the Radar repository: with all three wheels installed into
one disposable directory, do the two native extensions and the pure-Python
world contract stay in their own lanes?

The two evidences are NOT interchangeable, and the difference matters as soon
as a phase archives its wheels. ``wheel_smoke.py`` compares every shipped
``.py`` member against the checked-in bytes and REFUSES a wheel built before
the last source commit. This script deliberately does not: it validates
whichever three artifacts it is handed, so a wheel that has drifted from the
branch tip still passes here. Read a green run as "these three artifacts
coexist", never as "these three artifacts are current" - currency is
``wheel_smoke.py``'s claim, and an archived evidence set needs both.

Nine scenarios, one subprocess each, so a failure names itself instead of
collapsing a page of asserts into "the smoke failed":

=========  ==========================================================
scenario   what it proves
=========  ==========================================================
A          ``import witwin.core`` loads no Channel module, no ``rayd``,
           no ``drjit``, does not load the mesh-SDF CUDA package, and
           leaves CUDA uninitialized. Acceptance criterion A2, and the
           evidence that the mesh-SDF extension keeps an owner of its
           own rather than riding along with the world contract.
B          ``import witwin.radar`` loads no Channel module and does not
           import its own loader. The lazy ``__getattr__`` in
           ``witwin/radar/__init__.py`` is what makes A2 hold for the
           Radar package too, and a future eager import would break it
           silently.
C          ``import witwin.radar.propagation.channel_consumer`` DOES
           pull the Channel package - it is the adapter - but not the
           Channel native extension. Importing an adapter must not cost
           a native load.
D          ``witwin.radar.build_info()`` returns the full R-ADR-019
           record, ``origin == "packaged"``, and the binary it validated
           lives inside the disposable target.
E          ``witwin.channel.build_info()`` returns its ADR-006 record
           and the extension it loaded lives inside the target too.
           Together with D this is criterion A3 measured on installed
           artifacts rather than on a source checkout.
F          BOTH extensions loaded in one process: distinct files, both
           inside the target, disjoint ``torch.ops`` namespaces, and one
           real compute that crosses the boundary - Channel-native path
           delays fed into a Radar-native FMCW beat kernel.
G          With the packaged binary hidden, ``build_extension()`` RAISES.
           ``torch.utils.cpp_extension`` is never imported and no build
           directory appears. This is the no-silent-JIT half of A7 and
           the reason work item 1 exists.
H          Dependency closure: neither the base requirements nor the
           ``channel`` extra pulls a ray-tracing runtime. Criterion A8.
I          The installed CUDA sources re-hash to the ``source_fingerprint``
           the sidecar records, computed here WITHOUT calling the loader's
           own digest helper, so a wheel that repacked sources it did not
           build from is caught by an independent implementation.
=========  ==========================================================

Installation shape, and why it is a ``--target`` directory rather than a venv:

* ``pip install --disable-pip-version-check --no-deps --target <T>`` puts every
  installed file under one root, so ``Path(origin).is_relative_to(T)`` is a
  complete proof of where a module came from. A venv shares the interpreter's
  ``site-packages`` layout with the ambient environment and cannot answer that
  question as cleanly.
* the development environment this runs in has ``.pth`` files that prepend
  ``witwin``-providing source checkouts (Core, Radar and a raw
  ``_witwin_channel_src.pth`` for Channel) and editable finders for both. ``-I``
  does NOT suppress ``.pth`` processing, so each subprocess drops the editable
  finders by module name and drops every ``sys.path`` entry that provides a
  ``witwin`` directory other than the target. Anything the wheels do not ship
  is therefore unreachable by construction.
* ``rayd`` and ``drjit`` are deliberately left reachable. Scenarios A-C assert
  that they are NOT LOADED, and that claim is only interesting while loading
  them would have succeeded.

Note on ``PYTHONPATH``: ``python -I`` implies ``-E``, so an exported
``PYTHONPATH`` is ignored. The target is placed on ``sys.path`` from inside the
scenario script instead, which is the only form that actually takes effect.

Every wheel this validates is a locally built developer artifact
(``build_type == "developer"``): the local CUDA toolkit is newer than the
locked release toolchain, and the local architecture list is a subset of the
release set. Nothing here supports a release-build claim; that is a named
deferral.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


EVIDENCE_VERSION = 1

#: Runtimes that must never be loaded by, or required by, an installed Radar.
FORBIDDEN_RUNTIMES = ("rayd", "drjit", "mitsuba", "sionna")

#: Editable-install finders in this development environment. ``-I`` leaves them
#: registered, and each one would shadow the wheel it is named after.
_EDITABLE_FINDER_PREFIXES = (
    "_editable_impl_witwin",
    "_witwin_channel_editable",
    "__editable__",
)

#: Environment variables that would redirect either loader away from the
#: installed artifact. Scrubbed from every scenario subprocess: a smoke that
#: silently validated a developer-override binary would prove nothing about the
#: wheel.
_SCRUBBED_ENV_PREFIXES = ("WITWIN_RADAR_", "WITWIN_CHANNEL_", "RAYD_", "OPTIX_")

_RADAR_NATIVE_MEMBER = "witwin/radar/cuda/prebuilt/_radar_native.pyd"
_RADAR_NATIVE_MEMBERS = (
    _RADAR_NATIVE_MEMBER,
    "witwin/radar/cuda/prebuilt/_radar_native.so",
)

#: The nine translation units the Radar library is built from, as wheel members.
#: ``source_fingerprint`` hashes each file's NAME and content in this order.
_RADAR_SOURCE_MEMBERS = (
    "witwin/radar/cuda/extension.cpp",
    "witwin/radar/cuda/kernels/fmcw_beat.cu",
    "witwin/radar/cuda/kernels/frontend.cu",
    "witwin/radar/cuda/kernels/ofdm_cfr.cu",
    "witwin/radar/cuda/kernels/pulsed_echo.cu",
    "witwin/radar/cuda/kernels/scatter_response.cu",
    "witwin/radar/cuda/kernels/sensor_weight.cu",
    "witwin/radar/cuda/kernels/two_way_join.cu",
)

_REQUIRES_DIST = re.compile(r"^Requires-Dist:\s*(.+)$", re.MULTILINE)
_REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


class CoexistenceError(RuntimeError):
    """A scenario did not hold, or the harness could not run one."""


def _resolve_wheel(path: Path) -> Path:
    """Accept a wheel file, or a directory holding exactly one wheel.

    The directory form is what makes this callable from a CI tier, whose gate
    arguments are fixed strings and cannot know a platform-dependent wheel
    filename. "Exactly one" rather than "the newest": a directory with two
    wheels in it is an ambiguous input, and silently picking one is how a smoke
    ends up auditing last week's artifact. Both wheel smokes already resolve
    their inputs this way.
    """

    path = path.resolve()
    if path.is_dir():
        wheels = sorted(path.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(
                f"{path} must contain exactly one .whl file; found {len(wheels)}"
            )
        return wheels[0]
    if path.suffix != ".whl" or not path.is_file():
        raise ValueError(f"wheel does not exist: {path}")
    return path


def _preamble(target: Path) -> str:
    """Shared scenario prologue: isolate the target, then define ``emit``."""

    return f'''
import json
import sys
from pathlib import Path

TARGET = Path({str(target)!r}).resolve()

# ``-I`` still runs installed ``.pth`` files, so editable finders and raw
# source-checkout path entries survive into this process and would shadow the
# wheels. Drop the finders by owning module name...
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if not finder.__class__.__module__.startswith({_EDITABLE_FINDER_PREFIXES!r})
]
# ...and drop every path entry that offers a competing ``witwin`` portion.
# Namespace packages merge portions across sys.path, so leaving one in place
# would let a source checkout contribute submodules to the installed package.
sys.path[:] = [
    entry
    for entry in sys.path
    if not entry or not (Path(entry) / "witwin").is_dir()
]
sys.path.insert(0, str(TARGET))


def inside_target(path):
    return Path(path).resolve().is_relative_to(TARGET)


def loaded(prefix):
    """Modules currently imported under ``prefix`` (exact name or subpackage)."""
    return sorted(
        name
        for name in sys.modules
        if name == prefix or name.startswith(prefix + ".")
    )


def forbidden_loaded():
    hits = []
    for prefix in {FORBIDDEN_RUNTIMES!r}:
        hits.extend(loaded(prefix))
    return sorted(hits)


def emit(**fields):
    print(json.dumps(fields, sort_keys=True))
'''


def _scenario_a() -> str:
    return '''
import types

# Prove the detector fires before trusting it to report nothing. An empty
# "forbidden runtimes" list is only evidence if a loaded runtime would have
# shown up in it, and this is a scan over sys.modules, so it is cheap to check.
sys.modules["drjit"] = types.ModuleType("drjit")
sys.modules["rayd.torch"] = types.ModuleType("rayd.torch")
probe = forbidden_loaded()
del sys.modules["drjit"], sys.modules["rayd.torch"]
if probe != ["drjit", "rayd.torch"]:
    raise SystemExit(f"the forbidden-runtime detector does not work: {probe}")

import witwin.core

spec = witwin.core.__spec__
origin = spec.origin
if origin is None:
    origins = [str(Path(p).resolve()) for p in (spec.submodule_search_locations or [])]
else:
    origins = [str(Path(origin).resolve())]
outside = [entry for entry in origins if not inside_target(entry)]
if outside:
    raise SystemExit(f"witwin.core resolved outside the target: {outside}")

channel = loaded("witwin.channel")
if channel:
    raise SystemExit(f"importing witwin.core loaded Channel modules: {channel}")
runtimes = forbidden_loaded()
if runtimes:
    raise SystemExit(f"importing witwin.core loaded {runtimes}")
mesh_sdf = loaded("witwin.core.geometry.cuda")
if mesh_sdf:
    raise SystemExit(f"importing witwin.core loaded the mesh-SDF package: {mesh_sdf}")

torch = sys.modules.get("torch")
cuda_initialized = None if torch is None else bool(torch.cuda.is_initialized())
if cuda_initialized:
    raise SystemExit("importing witwin.core initialized CUDA")

emit(
    detector_probe=probe,
    core_origins=origins,
    channel_modules=channel,
    forbidden_modules=runtimes,
    mesh_sdf_modules=mesh_sdf,
    torch_imported=torch is not None,
    cuda_initialized=cuda_initialized,
)
'''


def _scenario_b() -> str:
    return '''
import witwin.radar

origin = Path(witwin.radar.__file__).resolve()
if not inside_target(origin):
    raise SystemExit(f"witwin.radar resolved outside the target: {origin}")

channel = loaded("witwin.channel")
if channel:
    raise SystemExit(f"importing witwin.radar loaded Channel modules: {channel}")
runtimes = forbidden_loaded()
if runtimes:
    raise SystemExit(f"importing witwin.radar loaded {runtimes}")
loader = [
    name
    for name in ("witwin.radar.cuda.build", "witwin.radar.cuda.identity")
    if name in sys.modules
]
if loader:
    raise SystemExit(f"importing witwin.radar loaded its native loader: {loader}")
if "torch.utils.cpp_extension" in sys.modules:
    raise SystemExit("importing witwin.radar imported the JIT compiler machinery")

emit(
    radar_origin=str(origin),
    channel_modules=channel,
    forbidden_modules=runtimes,
    cuda_loader_modules=loader,
)
'''


def _scenario_c() -> str:
    return '''
import witwin.radar.propagation.channel_consumer as adapter

origin = Path(adapter.__file__).resolve()
if not inside_target(origin):
    raise SystemExit(f"the adapter resolved outside the target: {origin}")

channel = loaded("witwin.channel")
if not channel:
    raise SystemExit("the Channel adapter imported no Channel module at all")
consumer = sys.modules.get("witwin.channel.propagation.consumer")
if consumer is None:
    raise SystemExit("the adapter did not import the Channel propagation consumer")
consumer_origin = Path(consumer.__file__).resolve()
if not inside_target(consumer_origin):
    raise SystemExit(f"the consumer resolved outside the target: {consumer_origin}")

native = loaded("witwin.channel._channel")
if native:
    raise SystemExit(f"importing the adapter loaded the Channel extension: {native}")
runtimes = forbidden_loaded()
if runtimes:
    raise SystemExit(f"importing the adapter loaded {runtimes}")

emit(
    adapter_origin=str(origin),
    consumer_origin=str(consumer_origin),
    channel_module_count=len(channel),
    channel_native_modules=native,
    forbidden_modules=runtimes,
)
'''


def _scenario_d() -> str:
    return '''
import witwin.radar

info = witwin.radar.build_info()
if info["origin"] != "packaged":
    raise SystemExit(f"installed radar extension origin is {info['origin']!r}")
extension_path = Path(info["extension_path"]).resolve()
if not inside_target(extension_path):
    raise SystemExit(f"radar extension loaded from outside the target: {extension_path}")
record = info["native_build"]
if not isinstance(record, dict):
    raise SystemExit("a packaged radar extension must carry a validated build record")
missing = [
    key
    for key in (
        "binary_sha256",
        "build_fingerprint",
        "build_type",
        "cuda_architectures",
        "extension_name",
        "operator_symbols",
        "radar_git_sha",
        "source_fingerprint",
        "torch_version",
    )
    if key not in record
]
if missing:
    raise SystemExit(f"radar build record is missing {missing}")
if record["extension_name"] != "_radar_native":
    raise SystemExit(f"unexpected extension name {record['extension_name']!r}")

emit(
    origin=info["origin"],
    extension_path=str(extension_path),
    radar_abi_version=info["radar_abi_version"],
    build_type=record["build_type"],
    build_fingerprint=record["build_fingerprint"],
    binary_sha256=record["binary_sha256"],
    source_fingerprint=record["source_fingerprint"],
    radar_git_sha=record["radar_git_sha"],
    cuda_architectures=record["cuda_architectures"],
    operator_count=len(record["operator_symbols"]),
)
'''


def _scenario_e() -> str:
    return '''
import witwin.channel

info = witwin.channel.build_info()
native = sys.modules.get("witwin.channel._channel")
if native is None:
    raise SystemExit("build_info() did not load the Channel extension")
native_origin = Path(native.__file__).resolve()
if not inside_target(native_origin):
    raise SystemExit(f"Channel extension loaded from outside the target: {native_origin}")
if info.get("backend") != "channel":
    raise SystemExit(f"unexpected Channel backend {info.get('backend')!r}")
if info.get("uses_dr_jit") is not False:
    raise SystemExit("the Channel wheel must report uses_dr_jit=false")
if info.get("uses_rayd_native") is not True:
    raise SystemExit("the Channel wheel must report uses_rayd_native=true")

from witwin.channel.propagation import consumer

emit(
    native_origin=str(native_origin),
    build_info_key_count=len(info),
    build_type=info["build_type"],
    build_fingerprint=info["build_fingerprint"],
    channel_abi_version=info["channel_abi_version"],
    material_abi_version=info["material_abi_version"],
    rayd_commit=info["rayd_commit"],
    consumer_contract_version=consumer.CONTRACT_VERSION,
)
'''


def _scenario_f() -> str:
    """Both extensions in one process, plus one compute that crosses them.

    The compute is deliberately end to end and small: a one-wall Core world,
    compiled by Channel, a frozen line-of-sight leg reevaluated through the
    Radar adapter, and the resulting NATIVE-produced delays fed straight into
    the Radar FMCW beat kernel. Two extensions, one tensor, no host round trip
    in between.
    """

    return '''
import math

import torch

def namespaces():
    return {name.split("::", 1)[0] for name in torch._C._dispatch_get_all_op_names()}

baseline = namespaces()

import witwin.channel
witwin.channel.build_info()
channel_native = sys.modules["witwin.channel._channel"]
after_channel = namespaces()
channel_namespaces = sorted(after_channel - baseline)

from witwin.radar.cuda import build as radar_build

radar = radar_build.build_extension()
after_radar = namespaces()
radar_namespaces = sorted(after_radar - after_channel)

channel_file = Path(channel_native.__file__).resolve()
radar_file = Path(radar.__file__).resolve()
if channel_file == radar_file:
    raise SystemExit("both extensions resolved to one file")
for label, path in (("channel", channel_file), ("radar", radar_file)):
    if not inside_target(path):
        raise SystemExit(f"{label} extension loaded from outside the target: {path}")
if radar_namespaces != ["_radar_native"]:
    raise SystemExit(f"radar registered {radar_namespaces}, expected ['_radar_native']")
overlap = sorted(set(channel_namespaces) & set(radar_namespaces))
if overlap:
    raise SystemExit(
        f"the two extensions register into the same dispatcher namespaces: {overlap}"
    )

# --- one compute that crosses both extensions -------------------------------
from witwin.core import AntennaState, Mesh, PhysicalMaterial, Scene, Structure
from witwin.core.identity import reserve_antenna_id
from witwin.channel.scene import compile as compile_scene
from witwin.radar.propagation import RadarEndpointSpec
from witwin.radar.propagation.channel_consumer import ChannelPropagationAdapter
from witwin.radar.synthesis.contracts import FmcwBeatSpec
from witwin.radar.synthesis.fmcw_beat import synthesize_beat_rows

REFERENCE_HZ = 77.0e9
TX = (0.0, 0.0, 0.0)
RX = (0.15, 0.0, 0.0)

# ``Mesh`` recentres authored vertices unless told not to, which would move the
# wall and quietly change every delay below.
mesh = Mesh(
    vertices=torch.tensor(
        ((3.0, -2.0, -2.0), (3.0, 2.0, -2.0), (3.0, 2.0, 2.0), (3.0, -2.0, 2.0)),
        dtype=torch.float32,
    ),
    faces=torch.tensor(((0, 1, 2), (0, 2, 3)), dtype=torch.int64),
    recenter=False,
    fill_mode="surface",
    topology_diagnostics=False,
)
scene = Scene(
    structures=(
        Structure(
            geometry=mesh,
            material=PhysicalMaterial(name="concrete", eps_r=5.24, sigma_e=0.0462),
            structure_id=1,
            material_id=1,
            assignment_id=1,
            surface_id=1,
        ),
    ),
    endpoints=[
        AntennaState(
            reserve_antenna_id(77001), "tx", torch.tensor(TX, dtype=torch.float32)
        )
    ],
)
compiled = compile_scene(scene, reference_frequency_hz=REFERENCE_HZ)


def endpoint(position, stable_id, power_w=None):
    return RadarEndpointSpec(
        stable_ids=torch.tensor([stable_id], dtype=torch.int64, device="cuda"),
        positions_m=torch.tensor([position], dtype=torch.float32, device="cuda"),
        polarizations=torch.tensor(
            [(0.0, 0.0, 1.0)], dtype=torch.float32, device="cuda"
        ),
        powers_w=(
            None
            if power_w is None
            else torch.full((1,), float(power_w), dtype=torch.float32, device="cuda")
        ),
    )


adapter = ChannelPropagationAdapter(
    compiled,
    reference_frequency_hz=REFERENCE_HZ,
    components=frozenset({"los"}),
    max_depth=0,
)
source = endpoint(TX, 10, power_w=1.0)
sink = endpoint(RX, 30)
frozen = adapter.freeze(source, sink)
leg = adapter.reevaluate(frozen, source, sink, ad_mode="none")

delay_s = leg.delay_s.reshape(-1)
if delay_s.device.type != "cuda":
    raise SystemExit(f"Channel published delays on {delay_s.device}, not CUDA")
row_count = int(delay_s.numel())
if row_count == 0:
    raise SystemExit("the crossing compute discovered no path to carry")

SAMPLE_RATE_HZ = 4.4e6
SLOPE_HZ_PER_S = 60.012e12
spec = FmcwBeatSpec(
    num_samples=32,
    num_chirps=2,
    sample_period_s=1.0 / SAMPLE_RATE_HZ,
    chirp_period_s=65.0e-6,
    slope_hz_per_s=SLOPE_HZ_PER_S,
    t_start_s=6.0e-6,
    reference_frequency_hz=REFERENCE_HZ,
    carrier_hz=0.0,
    carrier_rate_hz=REFERENCE_HZ,
)
delay_rate = torch.zeros_like(delay_s)
weight = torch.ones(row_count, dtype=torch.complex64, device="cuda")
offsets = torch.tensor([0, row_count], dtype=torch.int64, device="cuda")
cube = synthesize_beat_rows(delay_s, delay_rate, weight, offsets, spec)
if cube.shape != (spec.num_chirps, 1, spec.num_samples):
    raise SystemExit(f"unexpected cube shape {tuple(cube.shape)}")

# The crossing has to be checked NUMERICALLY, not by "the output is non-empty".
# With unit weights every sample has magnitude one, so total energy is a
# constant that would survive the Channel delay being dropped on the floor.
#
# Two checks that cannot: the Channel side must have produced the free-space
# delay for the authored geometry, and the Radar kernel's beat frequency must
# be the FMCW law evaluated at exactly that delay.
tau = float(delay_s[0])
expected_tau = 0.15 / 299792458.0
if abs(tau - expected_tau) > 1.0e-6 * expected_tau:
    raise SystemExit(
        f"Channel published delay {tau} for a 0.15 m line of sight, expected "
        f"{expected_tau}"
    )
chirp = cube[0, 0]
increment = float(torch.angle(chirp[1:] * chirp[:-1].conj()).mean())
predicted = 2.0 * math.pi * SLOPE_HZ_PER_S * tau / SAMPLE_RATE_HZ
if abs(increment - predicted) > 1.0e-5 * abs(predicted):
    raise SystemExit(
        f"beat phase increment {increment} does not match the FMCW law at the "
        f"Channel delay ({predicted}); the crossing tensor is not what the "
        "kernel consumed"
    )
perturbed = synthesize_beat_rows(
    delay_s * 1.01, delay_rate, weight, offsets, spec
)
if torch.allclose(cube, perturbed):
    raise SystemExit("the beat cube does not depend on the Channel delay at all")

emit(
    channel_extension=str(channel_file),
    radar_extension=str(radar_file),
    channel_dispatcher_namespaces=channel_namespaces,
    radar_dispatcher_namespaces=radar_namespaces,
    crossing_row_count=row_count,
    crossing_delay_s=tau,
    crossing_expected_delay_s=expected_tau,
    crossing_cube_shape=list(cube.shape),
    crossing_beat_phase_increment_rad=increment,
    crossing_predicted_phase_increment_rad=predicted,
    crossing_perturbed_max_abs_delta=float((cube - perturbed).abs().max()),
)
'''


def _scenario_g(temp_root: Path) -> str:
    """The packaged binary is hidden by the parent before this runs."""

    return f'''
import tempfile

build_root = Path(tempfile.gettempdir()).resolve() / "_radar_native"
before = (
    sorted(entry.name for entry in build_root.iterdir()) if build_root.is_dir() else []
)

from witwin.radar.cuda import build as radar_build
from witwin.radar.cuda.identity import RadarExtensionLoadError

packaged = radar_build.prebuilt_extension_path()
if packaged.exists():
    raise SystemExit(f"the packaged binary was supposed to be hidden: {{packaged}}")

try:
    radar_build.build_extension()
except RadarExtensionLoadError as exc:
    failure = str(exc)
else:
    raise SystemExit("a missing packaged binary did not fail the load")

if "torch.utils.cpp_extension" in sys.modules:
    raise SystemExit("the failing load imported the JIT compiler machinery")
after = (
    sorted(entry.name for entry in build_root.iterdir()) if build_root.is_dir() else []
)
if after != before:
    raise SystemExit(f"a build directory appeared under {{build_root}}: {{after}}")
for token in ("WITWIN_RADAR_NATIVE_DEVELOPER_OVERRIDE", "build_radar_cuda_prebuilt"):
    if token not in failure:
        raise SystemExit(f"the failure message does not name {{token!r}}: {{failure}}")

emit(
    packaged_path=str(packaged),
    error_type="RadarExtensionLoadError",
    error_message=failure,
    cpp_extension_imported=False,
    build_root={str(temp_root)!r},
    build_root_entries_before=before,
    build_root_entries_after=after,
)
'''


def _scenario_i(records: dict[str, str]) -> str:
    """Re-hash the installed sources with an independent implementation."""

    return f'''
import hashlib

expected = {records["source_fingerprint"]!r}
root = TARGET / "witwin" / "radar" / "cuda"
members = {list(_RADAR_SOURCE_MEMBERS)!r}
digest = hashlib.sha256()
sizes = {{}}
for member in members:
    path = TARGET / member
    if not path.is_file():
        raise SystemExit(f"the wheel did not install {{member}}")
    payload = path.read_bytes()
    sizes[member] = len(payload)
    # ``identity.source_digest``: file NAME, NUL, content, NUL - reimplemented
    # here so a defect in that helper cannot confirm itself.
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\\0")
    digest.update(payload)
    digest.update(b"\\0")
observed = digest.hexdigest()
if observed != expected:
    raise SystemExit(
        f"installed sources hash to {{observed}}, sidecar records {{expected}}"
    )

emit(
    source_fingerprint=observed,
    sidecar_source_fingerprint=expected,
    member_count=len(members),
    member_sizes=sizes,
)
'''


def _scenario_env(temp_root: Path) -> dict[str, str]:
    """A subprocess environment that cannot reach a non-wheel artifact."""

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_SCRUBBED_ENV_PREFIXES)
    }
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # ``-I`` ignores PYTHONPATH; it is exported only so an inherited value can
    # never be the reason a scenario passed. The scenario prologue is what
    # actually places the target on sys.path.
    env["PYTHONPATH"] = ""
    # A private temp root makes scenario G's "no build directory appeared"
    # assertion exact instead of a claim about a shared system directory.
    for name in ("TMP", "TEMP", "TMPDIR"):
        env[name] = str(temp_root)
    return env


def _run_scenario(
    *, name: str, code: str, target: Path, scratch: Path, temp_root: Path
) -> dict[str, object]:
    script = scratch / f"scenario_{name.lower()}.py"
    script.write_text(_preamble(target) + code, encoding="ascii")
    result = subprocess.run(
        [sys.executable, "-I", str(script)],
        cwd=str(target),
        env=_scenario_env(temp_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CoexistenceError(
            f"scenario {name} failed (exit {result.returncode})\n"
            f"{result.stdout.strip()}\n{result.stderr.strip()}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise CoexistenceError(
            f"scenario {name} printed {len(lines)} lines, expected exactly one "
            f"JSON object:\n{result.stdout}"
        )
    try:
        detail = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise CoexistenceError(f"scenario {name} did not emit JSON: {lines[0]}") from exc
    if not isinstance(detail, dict):
        raise CoexistenceError(f"scenario {name} emitted {type(detail).__name__}")
    return detail


def _wheel_native_record(radar_wheel: Path) -> dict[str, object]:
    with zipfile.ZipFile(radar_wheel) as archive:
        names = set(archive.namelist())
        member = next((name for name in _RADAR_NATIVE_MEMBERS if name in names), None)
        if member is None:
            raise CoexistenceError(
                f"{radar_wheel.name} ships no radar native member; "
                "it is not an artifact this smoke can validate"
            )
        sidecar = f"{member.rsplit('.', 1)[0]}.build-info.json"
        if sidecar not in names:
            raise CoexistenceError(f"{radar_wheel.name} ships no {sidecar}")
        return json.loads(archive.read(sidecar).decode("utf-8"))


def _wheel_metadata(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA") and name.count("/") == 1
        ]
        if len(names) != 1:
            raise CoexistenceError(
                f"{wheel.name} carries {len(names)} dist-info METADATA members"
            )
        return archive.read(names[0]).decode("utf-8")


def _requirement_names(metadata: str) -> list[str]:
    names = []
    for requirement in _REQUIRES_DIST.findall(metadata):
        match = _REQUIREMENT_NAME.match(requirement.strip())
        if match is None:
            raise CoexistenceError(f"unparseable requirement: {requirement!r}")
        names.append(match.group(1).lower().replace("_", "-"))
    return names


def _forbidden(names) -> list[str]:
    return sorted(
        name
        for name in names
        if any(
            name == runtime or name.startswith(runtime + "-")
            for runtime in FORBIDDEN_RUNTIMES
        )
    )


def _scenario_h(
    *, wheels: dict[str, Path], radar_wheel: Path, wheel_dir: Path, temp_root: Path
) -> dict[str, object]:
    """Dependency closure, declared and resolved.

    Two halves, because they answer different questions and have different
    blind spots.

    The DECLARED half is authoritative here. It reads ``Requires-Dist`` out of
    all three built wheels, extras included, and asserts that no requirement
    anywhere in the witwin closure names a ray-tracing runtime. It is exact,
    offline, and - crucially - it sees a dependency even when that distribution
    is already installed in the ambient environment.

    The RESOLVED half asks pip what installing ``witwin-radar[channel]`` would
    actually do, which is the check that catches an extra that silently does
    nothing. It runs WITHOUT ``--ignore-installed`` so it does not have to
    re-resolve Torch over the network, and that is exactly its blind spot: a
    forbidden distribution that happens to be installed already would be
    reported as satisfied rather than as an install. That is why the declared
    half, not this one, is the criterion A8 evidence.
    """

    declared: dict[str, list[str]] = {}
    for label, wheel in wheels.items():
        declared[label] = _requirement_names(_wheel_metadata(wheel))
    every = sorted({name for names in declared.values() for name in names})
    declared_forbidden = _forbidden(every)
    if declared_forbidden:
        raise CoexistenceError(
            f"the witwin wheels declare forbidden requirements: {declared_forbidden}"
        )
    if "witwin-channel" not in declared["radar"]:
        raise CoexistenceError(
            "the radar wheel declares no witwin-channel requirement at all, so "
            "'the radar wheel needs no ray tracer' would be true by omission"
        )

    report = temp_root / "dry-run-report.json"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--dry-run",
        "--report",
        str(report),
        "--find-links",
        str(wheel_dir),
        f"{radar_wheel}[channel]",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {
            "declared_requirements": declared,
            "declared_union": every,
            "declared_forbidden": declared_forbidden,
            "resolved": None,
            "resolved_forbidden": None,
            "resolver_status": "unresolved",
            "resolver_detail": result.stderr.strip().splitlines()[-3:]
            or ["(no stderr)"],
        }
    payload = json.loads(report.read_text(encoding="utf-8"))
    resolved = sorted(
        entry["metadata"]["name"].lower().replace("_", "-")
        for entry in payload.get("install", [])
    )
    resolved_forbidden = _forbidden(resolved)
    if resolved_forbidden:
        raise CoexistenceError(
            f"installing witwin-radar[channel] would pull {resolved_forbidden}"
        )
    if "witwin-channel" not in resolved:
        raise CoexistenceError(
            "the channel extra resolved without witwin-channel; the extra is "
            f"not doing anything: {resolved}"
        )
    return {
        "declared_requirements": declared,
        "declared_union": every,
        "declared_forbidden": declared_forbidden,
        "resolved": resolved,
        "resolved_forbidden": resolved_forbidden,
        "resolver_status": "resolved",
        "resolver_detail": [],
    }


def _install(target: Path, wheels: list[Path]) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--target",
        str(target),
        *(str(wheel) for wheel in wheels),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise CoexistenceError(
            f"installing the three wheels failed:\n{result.stdout}\n{result.stderr}"
        )


def _hide_radar_binary(target: Path) -> Path:
    binary = next(
        (
            target / member
            for member in _RADAR_NATIVE_MEMBERS
            if (target / member).is_file()
        ),
        None,
    )
    if binary is None:
        raise CoexistenceError("the installed target has no radar native binary to hide")
    hidden = binary.with_suffix(binary.suffix + ".hidden")
    binary.rename(hidden)
    return hidden


def run(
    *,
    core_wheel: Path,
    channel_wheel: Path,
    radar_wheel: Path,
    workspace: Path,
) -> dict[str, object]:
    target = workspace / "site-packages"
    scratch = workspace / "scenarios"
    temp_root = workspace / "temp"
    for directory in (scratch, temp_root):
        directory.mkdir(parents=True, exist_ok=True)

    record = _wheel_native_record(radar_wheel)
    _install(target, [core_wheel, channel_wheel, radar_wheel])

    scenarios: dict[str, object] = {}
    for name, code in (
        ("A", _scenario_a()),
        ("B", _scenario_b()),
        ("C", _scenario_c()),
        ("D", _scenario_d()),
        ("E", _scenario_e()),
        ("F", _scenario_f()),
        ("I", _scenario_i(record)),
    ):
        scenarios[name] = _run_scenario(
            name=name, code=code, target=target, scratch=scratch, temp_root=temp_root
        )

    scenarios["H"] = _scenario_h(
        wheels={"core": core_wheel, "channel": channel_wheel, "radar": radar_wheel},
        radar_wheel=radar_wheel,
        wheel_dir=radar_wheel.parent,
        temp_root=temp_root,
    )

    # G mutates the installed target, so it runs last and puts it back.
    hidden = _hide_radar_binary(target)
    try:
        scenarios["G"] = _run_scenario(
            name="G",
            code=_scenario_g(temp_root),
            target=target,
            scratch=scratch,
            temp_root=temp_root,
        )
    finally:
        hidden.rename(hidden.with_suffix(""))

    return {
        "evidence_version": EVIDENCE_VERSION,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "wheels": {
            "core": core_wheel.name,
            "channel": channel_wheel.name,
            "radar": radar_wheel.name,
        },
        "wheel_sizes_bytes": {
            "core": core_wheel.stat().st_size,
            "channel": channel_wheel.stat().st_size,
            "radar": radar_wheel.stat().st_size,
        },
        "radar_build_type": record["build_type"],
        "scenarios": {name: scenarios[name] for name in sorted(scenarios)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install the Core, Channel and Radar wheels into one disposable "
            "target and prove they coexist, import in isolation, report their "
            "build identity, and never silently JIT."
        )
    )
    parser.add_argument("--core-wheel", type=Path, required=True)
    parser.add_argument("--channel-wheel", type=Path, required=True)
    parser.add_argument("--radar-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workspace",
        type=Path,
        help=(
            "Reuse this directory instead of a temporary one. Only for "
            "debugging a failing scenario; it is not cleaned up."
        ),
    )
    args = parser.parse_args()

    wheels = {}
    for label, given in (
        ("core", args.core_wheel),
        ("channel", args.channel_wheel),
        ("radar", args.radar_wheel),
    ):
        try:
            wheels[label] = _resolve_wheel(given)
        except ValueError as error:
            parser.error(f"--{label}-wheel: {error}")

    if args.workspace is not None:
        workspace = args.workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.mkdtemp(prefix="radar-coexistence-")
        workspace = Path(cleanup)

    try:
        evidence = run(
            core_wheel=wheels["core"],
            channel_wheel=wheels["channel"],
            radar_wheel=wheels["radar"],
            workspace=workspace,
        )
    except CoexistenceError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)

    encoded = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
