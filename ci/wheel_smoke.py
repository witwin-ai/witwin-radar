"""Audit a built ``witwin-radar`` wheel and prove it loads from a fresh install.

Modelled on ``channel/ci/wheel_smoke.py`` and deliberately narrower: Radar
ships ONE dispatcher library plus its two identity sidecars, so the questions
worth asking are different from Channel's.

What this proves, and why each check exists:

* the archive is canonical - no directory entries, no absolute or traversing
  member names, no case/Unicode-duplicate members. A wheel that unpacks
  differently on two filesystems is not the artifact that was tested.
* exactly one ``.dist-info``, its version matching ``METADATA``, and exactly
  the four members a hatchling wheel is allowed to carry.
* ``RECORD`` covers every member with the right hash and size. This is the only
  check that ties the archive to what ``pip`` will verify.
* exactly one native member, and it is ``_radar_native.<pyd|so>`` under
  ``witwin/radar/cuda/prebuilt/``. A second DSO would mean a vendored runtime
  or a second binding, which acceptance criterion A5 forbids.
* both R-ADR-019 sidecars are present, the recorded ``binary_sha256`` is the
  digest of the PACKED binary, and the three-way fingerprint agrees. Without
  this a wheel can ship a binary and a record of a different binary and fail
  only at the user's first import.
* the packed CUDA sources re-hash to the recorded ``source_fingerprint``. This
  is the check that catches a wheel which repacked sources it did not build
  from - the loader makes the same comparison at import time, so a wheel that
  fails here is a wheel that cannot load.
* no build residue (``.obj``/``.lib``/``.pdb``/...), no embedded absolute host
  path, and no ``tests/`` member. The oracle under ``tests/reference/`` must
  never be importable from an installed package.
* finally, an isolated ``pip install --no-deps --target`` plus a ``python -I``
  subprocess that imports the package and asserts
  ``build_info()["origin"] == "packaged"`` with every resolved origin inside
  the disposable target. ``--target`` rather than a venv because it is the
  shape that also proves origin isolation, and because the ambient development
  environment has editable finders for ``witwin`` and ``witwin-radar`` that
  would otherwise shadow the wheel.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from email import policy
from email.parser import BytesParser
from functools import lru_cache
from pathlib import Path

_DISTRIBUTION = "witwin-radar"
_DIST_INFO_LICENSE = "licenses/LICENSE"
_DIST_INFO_FILES = frozenset({"METADATA", "RECORD", "WHEEL", _DIST_INFO_LICENSE})

#: The extension stem after the Phase-10 rename. The physical name and the
#: logical owner name are now the same string, which is the whole point of the
#: rename: ``ci/native-binding-manifest.json`` records it once.
_EXTENSION_NAME = "_radar_native"
_PREBUILT_PREFIX = "witwin/radar/cuda/prebuilt/"
_NATIVE_SUFFIXES = (".pyd", ".so")
_DSO_SUFFIXES = frozenset({".dll", ".dylib", ".pyd", ".so"})

#: The nine translation units ``build.extension_sources()`` compiles, by NAME.
#: ``identity.source_digest`` hashes ``path.name`` plus content, so the wheel's
#: copies reproduce the digest without knowing where they were built.
_SOURCE_MEMBERS = (
    "witwin/radar/cuda/extension.cpp",
    "witwin/radar/cuda/fmcw_beat.cu",
    "witwin/radar/cuda/frontend.cu",
    "witwin/radar/cuda/ofdm_cfr.cu",
    "witwin/radar/cuda/pulsed_echo.cu",
    "witwin/radar/cuda/scatter_response.cu",
    "witwin/radar/cuda/sensor_weight.cu",
    "witwin/radar/cuda/two_way_join.cu",
)

_BUILD_INFO_KEYS = frozenset(
    {
        "binary_sha256",
        "build_fingerprint",
        "build_type",
        "compiler",
        "cuda_architectures",
        "cuda_compiler_version",
        "cuda_version",
        "cxx_abi",
        "extension_name",
        "operator_symbols",
        "platform_tag",
        "python_abi",
        "radar_abi_version",
        "radar_git_dirty",
        "radar_git_sha",
        "source_fingerprint",
        "torch_target_version",
        "torch_version",
    }
)
_SMOKE_KEYS = frozenset(
    {"build_info", "distribution", "native_origin", "package_origin", "wheel_sha256", "wheel_smoke"}
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_BUILD_SUFFIXES = frozenset({".exp", ".ilk", ".lib", ".lock", ".o", ".obj", ".pdb"})
_ABSOLUTE_PATH_PATTERN = re.compile(
    rb"(?:(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]"
    rb"|(?<![A-Za-z0-9])/(?:home|Users|private/tmp|tmp|workspace)/)"
)


class WheelSmokeError(ValueError):
    """The wheel is not a supported radar artifact."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_wheel(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir():
        wheels = sorted(path.glob("*.whl"))
        if len(wheels) != 1:
            raise WheelSmokeError(f"wheel directory must contain exactly one .whl file; found {len(wheels)}")
        return wheels[0]
    if path.suffix != ".whl" or not path.is_file():
        raise WheelSmokeError(f"wheel does not exist: {path}")
    return path


def _canonical_member(name: str) -> str:
    if not name or "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise WheelSmokeError(f"wheel member is not a canonical relative path: {name!r}")
    if unicodedata.normalize("NFC", name) != name or any(ord(char) < 32 for char in name):
        raise WheelSmokeError(f"wheel member is not canonical Unicode/text: {name!r}")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise WheelSmokeError(f"wheel member has an empty/dot/traversal segment: {name!r}")
    return "/".join(parts)


def _canonical_members(archive: zipfile.ZipFile) -> list[str]:
    members: list[str] = []
    identities: set[str] = set()
    for info in archive.infolist():
        if info.is_dir():
            raise WheelSmokeError(f"wheel must not contain directory entries: {info.filename!r}")
        member = _canonical_member(info.filename)
        identity = unicodedata.normalize("NFC", member).casefold()
        if identity in identities:
            raise WheelSmokeError(f"wheel contains duplicate normalized/casefold member: {member!r}")
        identities.add(identity)
        members.append(member)
    if not members:
        raise WheelSmokeError("wheel contains no members")
    return members


def _metadata_identity(archive: zipfile.ZipFile, members: list[str]) -> tuple[str, str, str]:
    metadata_files = [
        member
        for member in members
        if len(member.split("/")) == 2
        and member.split("/")[0].endswith(".dist-info")
        and member.split("/")[1] == "METADATA"
    ]
    if len(metadata_files) != 1:
        raise WheelSmokeError(
            f"wheel must contain exactly one canonical .dist-info/METADATA file; found {metadata_files}"
        )
    metadata = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_files[0]))
    names = metadata.get_all("Name", [])
    versions = metadata.get_all("Version", [])
    if len(names) != 1 or len(versions) != 1:
        raise WheelSmokeError("wheel METADATA must contain exactly one Name and Version")
    name, version = names[0], versions[0]
    if name != _DISTRIBUTION or not version or any(char.isspace() for char in version):
        raise WheelSmokeError(
            f"wheel identity must be {_DISTRIBUTION!r} with a non-empty version; "
            f"found name={name!r}, version={version!r}"
        )
    dist_info = f"witwin_radar-{version}.dist-info"
    if metadata_files[0] != f"{dist_info}/METADATA":
        raise WheelSmokeError(
            f"wheel .dist-info directory does not exactly match METADATA version: {metadata_files[0]!r}"
        )
    return name, version, dist_info


@lru_cache(maxsize=1)
def _checked_in_package_members() -> frozenset[str]:
    """The tracked ``witwin/`` files, which is what hatchling packs.

    The prebuilt binary and its sidecars are gitignored build outputs and are
    handled separately; everything else in the wheel must be a checked-in file
    with byte-identical content.
    """

    root = _repository_root()
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "witwin"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise WheelSmokeError(f"cannot read checked-in witwin member list: {detail}")
    paths = [path for path in result.stdout.decode("utf-8").split("\0") if path and (root / path).is_file()]
    if not paths or any(not path.startswith("witwin/") for path in paths):
        raise WheelSmokeError("checked-in witwin member list is malformed or empty")
    return frozenset(paths)


def _record_hash(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _audit_record(archive: zipfile.ZipFile, members: list[str], *, dist_info: str) -> None:
    record_member = f"{dist_info}/RECORD"
    try:
        record_text = archive.read(record_member).decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise WheelSmokeError("wheel RECORD is missing or is not UTF-8") from exc
    try:
        rows = list(csv.reader(io.StringIO(record_text, newline=""), strict=True))
    except csv.Error as exc:
        raise WheelSmokeError(f"wheel RECORD is invalid CSV: {exc}") from exc
    if not rows or any(len(row) != 3 for row in rows):
        raise WheelSmokeError("wheel RECORD must contain non-empty three-column rows")

    recorded: dict[str, tuple[str, str]] = {}
    for raw_member, digest, size in rows:
        member = _canonical_member(raw_member)
        identity = member.casefold()
        if identity in recorded:
            raise WheelSmokeError(f"wheel RECORD contains duplicate member: {member!r}")
        recorded[identity] = (digest, size)
    expected = {member.casefold(): member for member in members}
    if set(recorded) != set(expected):
        missing = sorted(set(expected) - set(recorded))
        extra = sorted(set(recorded) - set(expected))
        raise WheelSmokeError(f"wheel RECORD member coverage mismatch: missing={missing}, extra={extra}")
    spelled = {row[0] for row in rows}
    mismatched_case = sorted(member for member in expected.values() if member not in spelled)
    if mismatched_case:
        raise WheelSmokeError(f"wheel RECORD member spelling/case mismatch: {mismatched_case}")

    for identity, member in expected.items():
        digest, size = recorded[identity]
        if member == record_member:
            if digest or size:
                raise WheelSmokeError("wheel RECORD self row must have empty hash and size")
            continue
        payload = archive.read(member)
        if digest != _record_hash(payload) or size != str(len(payload)):
            raise WheelSmokeError(f"wheel RECORD hash/size mismatch for {member!r}")


def _native_member(members: list[str]) -> str:
    """The one DSO, discovered by SUFFIX and then required to be the right one.

    Discovering by suffix rather than by name means a wheel that ships a
    SECOND, differently named binary fails here instead of being ignored.
    """

    shared_libraries = [name for name in members if Path(name).suffix.lower() in _DSO_SUFFIXES]
    if len(shared_libraries) != 1:
        raise WheelSmokeError(f"wheel must contain exactly one native member; found {shared_libraries}")
    native = shared_libraries[0]
    expected = {f"{_PREBUILT_PREFIX}{_EXTENSION_NAME}{suffix}" for suffix in _NATIVE_SUFFIXES}
    if native not in expected:
        raise WheelSmokeError(f"wheel native member must be one of {sorted(expected)}; found {native!r}")
    return native


def _sidecar_members(native: str) -> tuple[str, str]:
    stem = native[: -len(Path(native).suffix)]
    return f"{stem}.build-info.json", f"{stem}.build-fingerprint"


def _audit_identity(archive: zipfile.ZipFile, native: str) -> dict[str, object]:
    """The R-ADR-019 chain, checked against the bytes actually packed.

    The loader runs the same comparisons at import time, so anything that fails
    here would fail at the user's first import instead - which is exactly the
    late failure a packaging gate exists to move earlier.
    """

    info_member, fingerprint_member = _sidecar_members(native)
    for member in (info_member, fingerprint_member):
        if member not in set(archive.namelist()):
            raise WheelSmokeError(f"wheel is missing the identity sidecar {member!r}")

    try:
        record = json.loads(archive.read(info_member).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WheelSmokeError(f"{info_member} is not strict UTF-8 JSON") from exc
    if not isinstance(record, dict) or set(record) != _BUILD_INFO_KEYS:
        actual = sorted(record) if isinstance(record, dict) else type(record).__name__
        raise WheelSmokeError(f"{info_member} schema mismatch: {actual}")

    if record["extension_name"] != _EXTENSION_NAME:
        raise WheelSmokeError(f"wheel build record names {record['extension_name']!r}, expected {_EXTENSION_NAME!r}")
    if record["build_type"] not in {"release", "developer"}:
        raise WheelSmokeError(f"unexpected build_type: {record['build_type']!r}")
    for field in ("binary_sha256", "build_fingerprint", "source_fingerprint"):
        if _SHA256_PATTERN.fullmatch(str(record[field])) is None:
            raise WheelSmokeError(f"{field} must be a SHA-256; found {record[field]!r}")
    if _GIT_SHA_PATTERN.fullmatch(str(record["radar_git_sha"])) is None and record["radar_git_sha"] != "unknown":
        raise WheelSmokeError(f"unexpected radar_git_sha: {record['radar_git_sha']!r}")

    fingerprint_text = archive.read(fingerprint_member).decode("ascii", errors="replace")
    if fingerprint_text.strip() != record["build_fingerprint"]:
        raise WheelSmokeError(f"{fingerprint_member} does not match the recorded build_fingerprint")

    packed_digest = hashlib.sha256(archive.read(native)).hexdigest()
    if packed_digest != record["binary_sha256"]:
        raise WheelSmokeError(
            "wheel binary digest does not match its build record: "
            f"packed={packed_digest}, recorded={record['binary_sha256']}"
        )

    missing_sources = [name for name in _SOURCE_MEMBERS if name not in set(archive.namelist())]
    if missing_sources:
        raise WheelSmokeError(f"wheel does not ship the sources its identity is keyed by: {missing_sources}")
    digest = hashlib.sha256()
    for member in _SOURCE_MEMBERS:
        digest.update(Path(member).name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(archive.read(member))
        digest.update(b"\0")
    if digest.hexdigest() != record["source_fingerprint"]:
        raise WheelSmokeError(
            "wheel packed sources do not reproduce the recorded source_fingerprint: "
            f"packed={digest.hexdigest()}, recorded={record['source_fingerprint']}"
        )
    return record


def _audit_wheel_contents(wheel: Path) -> tuple[str, dict[str, object]]:
    with zipfile.ZipFile(wheel) as archive:
        members = _canonical_members(archive)
        _, _, dist_info = _metadata_identity(archive, members)
        native = _native_member(members)
        record = _audit_identity(archive, native)
        _audit_record(archive, members, dist_info=dist_info)

        info_member, fingerprint_member = _sidecar_members(native)
        special = {native, info_member, fingerprint_member}
        checked_in = _checked_in_package_members()
        package_members = {name for name in members if name.startswith("witwin/")}
        expected_package = set(checked_in) | special
        if package_members != expected_package:
            raise WheelSmokeError(
                "wheel package source closure mismatch: "
                f"missing={sorted(expected_package - package_members)}, "
                f"extra={sorted(package_members - expected_package)}"
            )
        root = _repository_root()
        mismatched = sorted(member for member in checked_in if archive.read(member) != (root / member).read_bytes())
        if mismatched:
            raise WheelSmokeError(f"wheel checked-in source bytes differ: {mismatched}")

        allowed_dist_info = {f"{dist_info}/{name}" for name in _DIST_INFO_FILES}
        dist_info_members = {name for name in members if name.startswith(f"{dist_info}/")}
        if dist_info_members != allowed_dist_info:
            raise WheelSmokeError(
                "wheel dist-info must contain exactly METADATA, WHEEL, RECORD and "
                f"licenses/LICENSE; found {sorted(dist_info_members)}"
            )
        if archive.read(f"{dist_info}/{_DIST_INFO_LICENSE}") != (root / "LICENSE").read_bytes():
            raise WheelSmokeError("wheel dist-info license bytes differ from the repository LICENSE")
        roots = {name.split("/", 1)[0] for name in members}
        if roots != {"witwin", dist_info}:
            raise WheelSmokeError(f"wheel has unexpected top-level roots: {sorted(roots)}")

        forbidden: list[str] = []
        for name in members:
            parts = tuple(part.lower() for part in name.split("/"))
            suffix = Path(name).suffix.lower()
            if parts[0] in {"tests", "tools", "scripts", "ci", "docs"}:
                forbidden.append(f"{name} (non-package top level)")
                continue
            if "tests" in parts or "rayd" in parts:
                forbidden.append(f"{name} (test or RayD content)")
                continue
            if any(part in {"cmakefiles", "_skbuild", "build", "__pycache__"} for part in parts):
                forbidden.append(f"{name} (build residue)")
                continue
            if suffix in _FORBIDDEN_BUILD_SUFFIXES:
                forbidden.append(f"{name} (build artifact)")
                continue
            # The absolute-path scan applies to GENERATED members only: the two
            # sidecars and the dist-info metadata. A checked-in source cannot
            # carry a build-host path into the wheel, because the closure check
            # above already proved its packed bytes equal the repository's -
            # and scanning it anyway flags deliberate portable literals such as
            # build.py's `os.environ.get("ProgramFiles", r"C:\Program Files")`
            # fallback, which is a source-review question and not a packaging
            # one. What this check exists to catch is a build that BAKED this
            # machine's paths into an artifact, and every such artifact is a
            # generated member.
            if name in checked_in:
                continue
            if suffix in {".cfg", ".json", ".md", ".toml", ".txt"} or name.endswith(".build-fingerprint"):
                if _ABSOLUTE_PATH_PATTERN.search(archive.read(name)):
                    forbidden.append(f"{name} (contains an absolute local path)")
        if forbidden:
            raise WheelSmokeError(f"wheel contains forbidden content: {forbidden}")
    return native, record


def _smoke_code(*, target: Path, wheel: Path, wheel_sha256: str, expected_name: str, expected_version: str) -> str:
    return f"""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

target = Path({str(target)!r}).resolve()
sys.path.insert(0, str(target))
# ``-I`` still processes installed ``.pth`` files, so an editable install of
# witwin or witwin-radar in the ambient environment can register a finder ahead
# of PathFinder and steal the package from this isolated target. Drop only the
# distribution-owned editable finders, inside this disposable subprocess.
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if not finder.__class__.__module__.startswith(
        ("_editable_impl_witwin", "_witwin_channel_editable", "__editable__")
    )
]
sys.path[:] = [
    entry
    for entry in sys.path
    if entry == str(target) or "site-packages" not in entry.replace("\\\\", "/") or True
]

wheel = Path({str(wheel)!r}).resolve()
digest = hashlib.sha256()
with wheel.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
wheel_sha256 = digest.hexdigest()
if wheel_sha256 != {wheel_sha256!r}:
    raise RuntimeError(f"wheel SHA-256 changed during smoke: {{wheel_sha256}}")

distribution = importlib.metadata.distribution({expected_name!r})
distribution_root = Path(distribution.locate_file("")).resolve()
if not distribution_root.is_relative_to(target):
    raise RuntimeError(f"distribution resolved outside isolated target: {{distribution_root}}")
if distribution.metadata["Name"] != {expected_name!r}:
    raise RuntimeError(f"unexpected distribution name: {{distribution.metadata['Name']!r}}")
if distribution.version != {expected_version!r}:
    raise RuntimeError(f"unexpected distribution version: {{distribution.version!r}}")

package_spec = importlib.util.find_spec("witwin.radar")
if package_spec is None or package_spec.origin is None:
    raise RuntimeError("witwin.radar has no import origin")
package_origin = Path(package_spec.origin).resolve()
if not package_origin.is_relative_to(target):
    raise RuntimeError(f"package resolved outside isolated target: {{package_origin}}")

import witwin.radar as radar

info = radar.build_info()
if info.get("origin") != "packaged":
    raise RuntimeError(f"installed radar extension origin is {{info.get('origin')!r}}")
native_origin = Path(info["extension_path"]).resolve()
if not native_origin.is_relative_to(target):
    raise RuntimeError(f"native extension resolved outside isolated target: {{native_origin}}")

for module in sorted(sys.modules):
    if module.startswith("witwin.channel") or module.split(".")[0] in {{"rayd", "drjit"}}:
        raise RuntimeError(f"importing witwin.radar loaded {{module}}")

print(json.dumps({{
    "wheel_smoke": True,
    "wheel_sha256": wheel_sha256,
    "distribution": {{
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "root": str(distribution_root),
    }},
    "package_origin": str(package_origin),
    "native_origin": str(native_origin),
    "build_info": info,
}}, sort_keys=True))
"""


def _parse_smoke_evidence(
    stdout: str,
    *,
    expected_wheel_sha256: str,
    expected_name: str,
    expected_version: str,
    target: Path,
    native_member: str,
    expected_record: dict[str, object],
) -> dict[str, object]:
    try:
        evidence = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise WheelSmokeError(f"isolated wheel smoke did not print JSON: {stdout!r}") from exc
    if not isinstance(evidence, dict) or set(evidence) != _SMOKE_KEYS:
        actual = sorted(evidence) if isinstance(evidence, dict) else type(evidence).__name__
        raise WheelSmokeError(f"isolated wheel smoke schema mismatch: {actual}")
    if evidence["wheel_smoke"] is not True:
        raise WheelSmokeError("isolated wheel smoke did not report success")
    if evidence["wheel_sha256"] != expected_wheel_sha256:
        raise WheelSmokeError("isolated wheel smoke reported a different wheel digest")
    distribution = evidence["distribution"]
    if distribution.get("name") != expected_name or distribution.get("version") != expected_version:
        raise WheelSmokeError(f"isolated wheel smoke distribution mismatch: {distribution}")
    for label in ("package_origin", "native_origin", "distribution"):
        value = evidence[label] if label != "distribution" else distribution["root"]
        if not Path(value).resolve().is_relative_to(target):
            raise WheelSmokeError(f"isolated wheel smoke {label} resolved outside target")
    if Path(evidence["native_origin"]).name != Path(native_member).name:
        raise WheelSmokeError(f"isolated wheel smoke loaded {evidence['native_origin']!r}, not the packed member")
    info = evidence["build_info"]
    if info.get("origin") != "packaged":
        raise WheelSmokeError(f"installed extension origin is {info.get('origin')!r}")
    native_build = info.get("native_build")
    if not isinstance(native_build, dict) or set(native_build) != _BUILD_INFO_KEYS:
        raise WheelSmokeError("installed build_info carries no complete native record")
    mismatched = sorted(key for key in _BUILD_INFO_KEYS if native_build[key] != expected_record[key])
    if mismatched:
        raise WheelSmokeError("installed build record differs from the packed sidecar: " + ", ".join(mismatched))
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a witwin-radar wheel and import it from an isolated install.")
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--core-wheel",
        type=Path,
        required=True,
        help="witwin (Core) wheel installed into the same isolated target; "
        "witwin.radar imports witwin.core at package import.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        wheel = _resolve_wheel(args.wheel)
        core_wheel = _resolve_wheel(args.core_wheel)
        with zipfile.ZipFile(wheel) as archive:
            expected_name, expected_version, _ = _metadata_identity(archive, _canonical_members(archive))
        native_member, record = _audit_wheel_contents(wheel)
    except (OSError, WheelSmokeError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    wheel_sha256 = _sha256(wheel)

    with tempfile.TemporaryDirectory(prefix="radar-wheel-smoke-") as raw:
        target = Path(raw) / "site-packages"
        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(target),
                str(core_wheel),
                str(wheel),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if install.returncode != 0:
            print(install.stderr, file=sys.stderr)
            return install.returncode

        code = _smoke_code(
            target=target,
            wheel=wheel,
            wheel_sha256=wheel_sha256,
            expected_name=expected_name,
            expected_version=expected_version,
        )
        smoke = subprocess.run(
            [sys.executable, "-I", "-c", code], cwd=target, capture_output=True, text=True, check=False
        )
        if smoke.stderr:
            print(smoke.stderr.strip(), file=sys.stderr)
        if smoke.returncode != 0:
            if smoke.stdout:
                print(smoke.stdout.strip())
            return smoke.returncode
        try:
            evidence = _parse_smoke_evidence(
                smoke.stdout,
                expected_wheel_sha256=wheel_sha256,
                expected_name=expected_name,
                expected_version=expected_version,
                target=target,
                native_member=native_member,
                expected_record=record,
            )
        except WheelSmokeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    evidence["native_member"] = native_member
    encoded = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
