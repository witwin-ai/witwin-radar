# WiTwin Radar - Differentiable Radar Simulator

A GPU-accelerated, differentiable FMCW radar simulator for generating synthetic radar data from 3D scenes. Propagation is consumed from the Channel propagation consumer, and waveform synthesis, path composition, and signal generation run in Radar's own CUDA kernels.

This module is derived from [RF-Genesis](https://github.com/Asixa/RF-Genesis).

## Get Started

CPython 3.10-3.14 and PyTorch 2.10 or newer are supported. Radar simulation runs in the `_radar_native` CUDA kernels and requires an NVIDIA GPU with CUDA. CPU construction remains useful for configuration and non-rendering helper workflows.
This package depends on the base `witwin` package.

Linux and Windows are supported targets. Release wheels include prebuilt native CUDA extensions for supported Python/platform combinations. Source builds require a CUDA-enabled PyTorch build, NVIDIA driver, CUDA toolkit, `ninja`, and a working C++ compiler.

Release wheels are built with CUDA 12.8 and contain one Python-independent native library using the LibTorch Stable ABI introduced for this surface in PyTorch 2.10. The same wheel and native binary are CI load-tested with PyTorch 2.10/cu128, 2.11/cu128, and 2.12/cu126 across CPython 3.10-3.14; no Torch-minor-specific binary selection or JIT rebuild is required. Their fat binaries contain native code for compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 10.1, and 12.0, plus compute 12.0 PTX. This includes native RTX 2080-class Turing and current data-center and RTX/RTX PRO Blackwell coverage. Linux wheels target `manylinux_2_35_x86_64`; the installed NVIDIA driver must support the CUDA 12.x runtime supplied by PyTorch.

For full CUDA 12.8 and Blackwell support, use at least driver 570.26 on Linux or 570.65 on Windows. Pre-Blackwell systems can use NVIDIA's CUDA 12.x minor-version compatibility floor (525.60.13 on Linux or 528.33 on Windows), subject to NVIDIA's compatibility-mode feature limits.

```bash
pip install witwin[radar]
```

## Quick Start

`Radar.simulate` is the entry point: a `witwin.core.Scene`, a list of frame
instants, and a scatter response in, a `[frame, TX, RX, slow, fast]` cube out.
`docs/pipeline_guide.md` walks the whole route; `examples/single_point.py` is
this snippet with its closed-form checks attached.

```python
import torch

from witwin.core import Scene
from witwin.radar import Radar, RadarConfig, ScatterSitePolicy
from witwin.radar.processing import ProcessingAxes, range_doppler, range_profile
from witwin.radar.scattering import ScalarRcsResponse

# FMCW radar configuration.
config = {
    "num_tx": 3,
    "num_rx": 4,
    "fc": 77e9,
    "slope": 60.012,
    "adc_samples": 256,
    "adc_start_time": 6,
    "sample_rate": 4400,
    "idle_time": 7,
    "ramp_end_time": 65,
    "chirp_per_frame": 128,
    "frame_per_second": 10,
    "num_doppler_bins": 128,
    "num_range_bins": 256,
    "num_angle_bins": 64,
    "power": 15,
    "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
    "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]],
}

radar = Radar(
    RadarConfig.from_dict(config),
    device="cuda",
    position=(0.0, 0.0, 0.0),
    target=(1.0, 0.0, 0.0),   # boresight along +x
)

# An empty world is legal: a scatter site is a DECLARED endpoint, not geometry.
scene = Scene(structures=())
sites = ScatterSitePolicy.explicit(
    torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float32, device=radar.device)
)
response = ScalarRcsResponse.from_rcs(
    1.0, reference_frequency_hz=config["fc"], device=radar.device
)

result = radar.simulate(scene, times=(0.0, 0.1, 0.2), response=response, sites=sites)
assert result.cube.shape[0] == 3          # [frame, TX, RX, chirp, sample]

# Post-processing is PyTorch and reads one metadata record.
from witwin.radar.synthesis import SlowTimeMode

axes = ProcessingAxes.from_synthesis(
    radar.synthesize(
        radar.last_radar_paths,
        slow_time_mode=SlowTimeMode.FROZEN_WEIGHT_WITH_CARRIER_RATE,
    ),
    radar.system_config.waveform_spec(),
    radar.system_config.sensors.array,
)
profile = range_profile(result.cube[0], axes=axes)
rd = range_doppler(profile)
```

Two conventions cost more time than anything else in this API, so they are
stated here rather than discovered:

* the endpoint polarization must be TRANSVERSE to the look direction. The
  default is `(0, 0, 1)`, which is transverse for a radar looking along `x` and
  PARALLEL for one looking along `-z`; a parallel polarization publishes an
  exactly zero transport and raises nothing. Pass `polarization=(0, 1, 0)` for a
  `-z` boresight.
* `witwin.core.Mesh` defaults `recenter=True` and silently subtracts the
  bounding-box centre from authored vertices. Always pass `recenter=False` for
  world-frame geometry.

## Scenes And Motion

The logical world is owned by `witwin.core`, not by Radar. There is no
radar-owned `Scene`, `SceneModule`, `Timeline` or `TransformMotion`: each of
those names now raises with a message naming its Core replacement.

```python
import torch

from witwin.core import Mesh, PhysicalMaterial, Scene, Structure
from witwin.core.dynamics import DynamicScene, LinearTrajectory

wall = Structure(
    geometry=Mesh(
        vertices=wall_vertices,
        faces=wall_faces,
        recenter=False,          # MANDATORY for world coordinates
        fill_mode="surface",
        topology_diagnostics=False,
    ),
    material=PhysicalMaterial(name="concrete", eps_r=5.24, sigma_e=0.0462),
    structure_id=1, material_id=1, assignment_id=1, surface_id=1,
)
scene = Scene(structures=(wall,))

# A moving world is a DynamicScene; `simulate` samples it at every frame instant.
moving = DynamicScene(
    scene,
    structure_trajectories={
        1: LinearTrajectory(
            origin=torch.zeros(3),
            velocity=torch.tensor((0.0, 0.0, 1.0)),
        )
    },
)
result = radar.simulate(moving, times=(0.0, 0.1, 0.2), response=response, sites=sites)
```

`Scene` is immutable and builds with `with_structures(...)`,
`with_endpoints(...)`, `with_material(...)` and `with_structure_geometry(...)`
rather than with mutating `add_*` methods.

After a call, five typed diagnostics describe the last frame - on the result and
on the radar, which are the same objects:

| name | type |
| --- | --- |
| `last_snapshot` | `witwin.core.SceneSnapshot` |
| `last_compiled_scene` | Channel `CompiledScene` |
| `last_propagation` | `witwin.radar.RadarPropagationLegs` |
| `last_radar_paths` | `witwin.radar.paths.RadarPathBatch` |
| `last_result` | `witwin.radar.RadarSimulationResult` |

Migrating from the pre-cutover API (`Radar.mimo`, `mimo_from_trace`,
`MimoPathCache`, `TraceResult`, the radar `Scene`, the Dirichlet solver):
`docs/dev/migration/phase11-cutover-migration-note.md` lists every break with
its replacement.

## Features

- One scene-driven entry point, `Radar.simulate`, from a `witwin.core.Scene` to
  a `[frame, TX, RX, slow, fast]` cube, with five typed diagnostics
- Native CUDA kernels for the two-way join, FMCW / OFDM / pulsed synthesis, the
  aspect-dependent scatter response, the sensor weight and the receive chain,
  each with an analytical backward and JVP companion
- Differentiable multipath propagation (line of sight and reflection) consumed
  from the Channel propagation consumer, with a fixed-topology freeze /
  reevaluate split; mesh vertices, permittivity, endpoint and site positions and
  the target cross section all reach the cube in both AD modes
- Shared-core geometry, structure and material primitives; SMPL body support
  through `witwin.radar.SMPLBody` as a Core `Structure` geometry
- Per-structure rigid motion with parent inheritance, declared on a
  `witwin.core.DynamicScene` and sampled per frame
- One processing facade, `witwin.radar.processing`: range profile, Range-Doppler,
  beam/range/velocity cube, AoA, beamforming, CFAR, point cloud and detection
  handoff, all PyTorch, sharing one `ProcessingAxes` metadata/axes/units record
  across FMCW, OFDM and pulsed. `witwin.radar.sigproc` keeps its public names as
  migration adapters
- The vendor DSP primitive surface is frozen and asserted by equality; the
  native-DSP gate was measured and the recorded answer is no native DSP
  (`tools/benchmark_processing.py`, `PERFORMANCE.md`)
- Tensor-first DSP outputs with backwards-compatible NumPy wrappers
- Optional antenna-pattern configuration, applied to a composed round trip
  through `Radar.simulate(..., antenna_pattern=...)`; the receive chain is the
  `frontend` configuration block

## Running Tests

```bash
cd radar
pytest tests/
pytest tests/ --gpu
```

## Examples

Run the maintained Python examples from the `radar/` root. Each has a notebook
twin beside it with the same content.

```bash
python -m examples.single_point
python -m examples.music_imaging
python -m examples.rgbd_range_doppler --input path/to/depths.npz
```

- `single_point` - one point target in front of a concrete wall: the target peak
  and the image-source multipath peak, both checked against closed forms.
- `music_imaging` - a 20 x 20 UPA resolving two targets in azimuth, checked
  against the analytic bearings.
- `rgbd_range_doppler` - a depth sequence turned into scatter sites, one
  Range-Doppler map per frame. It reads the `.npz` written by
  `examples/preprocess_rfgen_rd.py`, which converts `.npy`/`.npz` depth or
  point-cloud sequences and Azure Kinect `.mkv` files (with `pykinect_azure`
  installed). No depth asset ships with the repository; a missing `--input`
  raises and names that script.

All three require CUDA. There are no SMPL or AMASS examples in this tree.

## Installation

Python 3.10+ is required. Install a CUDA-enabled PyTorch build for simulation and tracing. Linux and Windows are supported; source builds require the NVIDIA driver, CUDA toolkit, `ninja`, and a C/C++ compiler on `PATH`.

```bash
pip install witwin[radar]
```

Core dependencies include `torch`, `numpy`, `scipy`, `tqdm`, and `matplotlib`. Propagation additionally requires `witwin-channel`; its release pin is provisional while the artifacts are being built, so it is consumed from a source checkout for now.

## Citation

If this module or its original RF-Genesis work is relevant to your research, please cite:

```bibtex
@inproceedings{chen2023rfgenesis,
  author = {Chen, Xingyu and Zhang, Xinyu},
  title = {RF Genesis: Zero-Shot Generalization of mmWave Sensing through Simulation-Based Data Synthesis and Generative Diffusion Models},
  booktitle = {ACM Conference on Embedded Networked Sensor Systems (SenSys '23)},
  year = {2023},
  pages = {1-14},
  address = {Istanbul, Turkiye},
  publisher = {ACM, New York, NY, USA},
  url = {https://doi.org/10.1145/3625687.3625798},
  doi = {10.1145/3625687.3625798}
}
```

## License

Witwin Radar is available under a dual-license model for academic and
non-commercial research use or commercial and enterprise use. See the
[Witwin licensing page](https://witwin.ai/license) for the applicable terms.

## Developer

<a href="http://xingyuchen.me/">
  <img src="https://github.com/Asixa.png" alt="Xingyu Chen" width="48" height="48" style="border-radius:50%;">
</a>

[Xingyu Chen](http://xingyuchen.me/)
