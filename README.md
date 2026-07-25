# WiTwin Radar - Differentiable Radar Simulator

A GPU-accelerated, differentiable FMCW radar simulator for generating synthetic radar data from 3D scenes. Propagation is consumed from the Channel propagation consumer, and waveform synthesis, path composition, and signal generation run in Radar's own CUDA kernels.

This module is derived from [RF-Genesis](https://github.com/Asixa/RF-Genesis).

## Get Started

CPython 3.10-3.14 and PyTorch 2.10 or newer are supported. Radar simulation uses the native Dirichlet CUDA backend and requires an NVIDIA GPU with CUDA. CPU construction remains useful for configuration and non-rendering helper workflows.
This package depends on the base `witwin` package.

Linux and Windows are supported targets. Release wheels include prebuilt native CUDA extensions for supported Python/platform combinations. Source builds require a CUDA-enabled PyTorch build, NVIDIA driver, CUDA toolkit, `ninja`, and a working C++ compiler.

Release wheels are built with CUDA 12.8 and contain one Python-independent native library using the LibTorch Stable ABI introduced for this surface in PyTorch 2.10. The same wheel and native binary are CI load-tested with PyTorch 2.10/cu128, 2.11/cu128, and 2.12/cu126 across CPython 3.10-3.14; no Torch-minor-specific binary selection or JIT rebuild is required. Their fat binaries contain native code for compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, 10.1, and 12.0, plus compute 12.0 PTX. This includes native RTX 2080-class Turing and current data-center and RTX/RTX PRO Blackwell coverage. Linux wheels target `manylinux_2_35_x86_64`; the installed NVIDIA driver must support the CUDA 12.x runtime supplied by PyTorch.

For full CUDA 12.8 and Blackwell support, use at least driver 570.26 on Linux or 570.65 on Windows. Pre-Blackwell systems can use NVIDIA's CUDA 12.x minor-version compatibility floor (525.60.13 on Linux or 528.33 on Windows), subject to NVIDIA's compatibility-mode feature limits.

```bash
pip install witwin[radar]
```

## Quick Start

```python
import numpy as np
import torch

from witwin.radar import Radar, RadarConfig
from witwin.radar.sigproc import process_pc, process_pc_tensor, process_rd, process_rd_tensor

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

# Use the native CUDA solver.
radar = Radar(
    RadarConfig.from_dict(config),
    device="cuda",
    position=(0.0, 0.0, 0.0),
    target=(0.0, 0.0, -5.0),
    fov=60.0,
)

point = np.array([[0.0, 0.0, -3.0]], dtype=np.float32)
velocity = np.array([[0.0, 0.0, 0.01]], dtype=np.float32)


def interp(t):
    # Return target intensity and position at time t.
    positions = torch.tensor(point + velocity * t, dtype=torch.float32, device=radar.device)
    intensities = torch.ones((positions.shape[0],), dtype=torch.float32, device=radar.device)
    return intensities, positions


# Simulate one frame, then extract point cloud and RD map.
frame = radar.mimo(interp, t0=0)
pc = process_pc(radar, frame)
rd, _, ranges, vels = process_rd(radar, frame)

# Tensor-first variants keep outputs on the frame device for real-time pipelines.
pc_gpu = process_pc_tensor(radar, frame)
rd_gpu, _, ranges_gpu, vels_gpu = process_rd_tensor(radar, frame)
```

## Scene API

Use `Radar(..., position=..., target=..., fov=...)` to define the radar pose, and `Scene.add_*` methods for scene assembly.

```python
from witwin.core import Material, Structure
from witwin.radar import Radar, RadarConfig, Scene, TransformMotion

radar = Radar(
    RadarConfig.from_dict(config),
    device="cuda",
    position=(0.0, 0.0, 0.0),
    target=(0.0, 0.0, -1.0),
    fov=60.0,
)
scene = Scene(device="cpu")

scene.add_structure(
    Structure(
        name="car_body",
        geometry=car_body_mesh,
        material=Material(eps_r=3.0),
    )
)
scene.add_mesh(name="wheel_fl", vertices=wheel_vertices, faces=wheel_faces, dynamic=True)
scene.add_structure_motion(
    "wheel_fl",
    TransformMotion(
        axis=(0.0, 1.0, 0.0),
        angular_velocity=32.0,
        origin=(0.0, 0.0, 0.0),
        space="local",
    ),
)

```

`Radar.simulate(scene, ...)` and `Radar.simulate_group(...)` have been REMOVED
along with the Dr.Jit ray tracer that backed them. Propagation now goes through
the Channel propagation consumer:

1. compile the scene and build a
   `witwin.radar.propagation.ChannelPropagationAdapter`;
2. `freeze` each leg once, outside the per-frame loop;
3. `reevaluate` it per frame at the new endpoint positions;
4. compose the legs with `witwin.radar.paths.TwoWayComposer` (radar source ->
   scatter site -> radar sink) or `DirectComposer` (source straight to sink);
5. synthesize with `witwin.radar.synthesis.synthesize_fmcw_beat`.

A scene-driven entry point that assembles those steps for a whole `Scene` does
not exist yet. `Radar.mimo`, `mimo_from_trace`, `mimo_from_paths`,
`path_cache_from_trace`, `chirp`, and `frame` are unaffected.

Available mutating scene methods:

- `Scene.add_structure(...)`
- `Scene.add_mesh(...)`
- `Scene.add_smpl(...)`
- `Scene.add_structure_motion(...)`
- `Scene.update_structure(...)`
- `Scene.remove(...)`

## Features

- Native Dirichlet CUDA kernels for chirp, frame, and MIMO generation
- Native CUDA autograd kernels for distance and amplitude gradients
- Differentiable multipath propagation (line of sight and reflection) consumed from the Channel propagation consumer, with a fixed-topology freeze/reevaluate split
- Shared-core geometry and structure primitives
- SMPL body support through `Scene.add_smpl(...)`
- Optional per-structure rigid motion with parent inheritance
- Torch-native DSP pipeline for range/Doppler processing and point-cloud extraction
- Tensor-first DSP outputs with backwards-compatible NumPy wrappers
- Optional antenna pattern, polarization, noise-model, and receiver-chain configuration

## Running Tests

```bash
cd radar
pytest tests/
pytest tests/ --gpu
```

## Examples

Run the maintained Python examples from the `radar/` root:

```bash
python -m examples.single_point
python -m examples.mesh_scene
python -m examples.humanbody
python -m examples.music_imaging
python -m examples.amass_pointcloud
python -m examples.gen_amass_video
python -m examples.rgbd_range_doppler --input path/to/depths.npy
```

`amass_pointcloud` and `gen_amass_video` additionally require AMASS BMLmovi data under `data/BMLmovi_full/BMLmovi/`. The rendering examples require RayD and CUDA; the SMPL examples also require `models/smpl_models/`.
`rgbd_range_doppler` reads `.npy`/`.npz` depth or point-cloud sequences, and can read Azure Kinect `.mkv` files when `pykinect_azure` is installed. It assumes the depth camera view is the radar view by default.

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
