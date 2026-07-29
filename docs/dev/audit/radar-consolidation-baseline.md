# Radar concept-axis consolidation baseline

Status: Phase 0 complete; Phase G active.

Recorded: 2026-07-28.

## Source state

| Repository | Commit | Execution source | Mutable source worktree |
| --- | --- | --- | --- |
| Radar | `ff2d9cc8efab487a4b978a1afdaab56994fac9aa` | repository worktree, clean before execution | this worktree now contains Phase 0/G changes |
| Channel | `c07b489e8ed100de8a03d4a46aae46a23899eed5` | clean no-hardlink clone `C:\tmp\radar-channel-git-c07b489` | concurrent edits exist and are untouched |
| Core | `7791ce21a23d471bf4306b21d6919000ef97bccc` | clean export `C:\tmp\radar-core-7791ce21` | existing user edits exist and are untouched |
| RayD | `94cf6eaf39f3625af482bb3fd8cba1377a804ecc` | clean no-hardlink clone `C:\tmp\radar-rayd-git-94cf6eaf` | source worktree is untouched |

The mutable sibling worktrees are not execution dependencies. Their exact HEAD
commits were exported or checked out into the clean snapshots above. Radar does
not modify the sibling source worktrees. Final acceptance uses the same commits
from clean checkouts or fresh exports; it never depends on uncommitted sibling
files.

## Runtime and native identity

- Python: 3.11.14 from `C:\Users\Asixa\miniconda3\envs\witwin2`
- Torch: 2.10.0; CUDA runtime recorded by Channel: 12.8
- GPU: NVIDIA GeForce RTX 5080, SM120
- Core version: 0.4.0
- Core Git tree: `274d6b9ee702660c4cffe09a3fc915e175ed9cb3`
- Channel version: 0.4.0
- Channel wheel SHA-256:
  `341e7c8fa5486985c4d4e0c093ca736bb001b0188c427abc4eda520283094681`
- Channel native fingerprint:
  `2dd9d779aa28307a2ba111ee823aa1a70eb6dcd1581801d2a94e480ea77aad7b`
- Channel build identity: clean `c07b489e`, Release, MSVC 19.44,
  `120-real;120-virtual`, Torch 2.10.0, Channel ABI 1, material ABI 3
- RayD build identity: clean `94cf6eaf`, source-linked, integration header
  SHA-256 `57f83ea460e376166fd5ee22a8243a7c1576a290e1de99c0cbe8e86e93392e14`
- Radar native fingerprint:
  `23d7d2db8271b3ee68a3d6e54f934f150c622f4f15d1093cf1e8409226cd8700`
- Runtime import roots:
  `C:\tmp\radar-channel-installed-c07b489-2dd9d779`,
  `C:\tmp\radar-core-7791ce21`,
  `E:\Code\witwin-platform\radar`

Core has no packaged, identity-bearing native library on Radar's main execution
path. Its immutable identity is therefore the clean commit plus Git tree rather
than a fabricated build fingerprint. Its optional mesh-SDF JIT is outside the
Channel-to-Radar baseline used here.

The Channel wheel was built from the clean fixed commit against the locked RayD
commit and imported successfully from an isolated target directory. Its embedded
build record reports `channel_git_dirty=false` and `rayd_dirty=false`.

## Test and numerical baseline

The clean snapshots collected **1579 tests** with no collection error and the
same node count as the mutable sibling worktrees.

The first quick tier, before installing the fixed Channel wheel, completed with
**792 passed, 793 skipped**, 55 warnings and 65% coverage. That result is retained
as evidence of GOV-024, not accepted as integration evidence.

With the fixed Channel wheel installed in the isolated import root:

- quick CPU suite: **796 passed, 793 skipped**, 55 warnings, 65% coverage;
- GPU suite: **1587 passed, 2 skipped**, 62 warnings, 83% coverage;
- loader contract: **22 passed**;
- extension boundary: passed against packaged `_radar_native.pyd`;
- all pre-existing static gates: passed.

The two GPU skips are not the Channel main chain:

1. `tests/test_phase10_boundary.py` requires the optional external
   `WITWIN_CHANNEL_EXTENSION_PATH` developer override; the packaged Channel wheel
   was imported and fingerprinted instead.
2. `tests/test_phase10_coexistence_smoke.py` requires a nightly-only coexistence
   evidence artifact that is intentionally absent from this checkout.

The plan's numerical smoke, `python -m examples.single_point`, passed:

- cube `(3, 3, 4, 128, 256)`;
- one scene compile and one path discovery;
- 48 composed rows over 12 pairs;
- measured `|C_rt| = 1.726919e-06` versus closed-form
  `1.726922e-06`;
- range peak 3.0051 m for a 3.0000 m target;
- multipath peak near 5.0000 m;
- 40 point-cloud points.

The complete tier logs are:

- `C:\tmp\radar-cuda-baseline-019fab10.out`
- `C:\tmp\radar-cuda-baseline-019fab10.err`

Phase 0 is closed by this record. Production deletion/movement still waits for
Phase G's manifests, planted gate calibrations and adversarial before-evidence to
be staged together.

Phase 1 intentionally deletes compatibility-only tests. Each removed test file
is listed in the governance inventory. The post-reset collection becomes the
active node baseline; later layout phases may not silently shrink it.
