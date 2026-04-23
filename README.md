# WiTwin Radar - Differentiable Radar Simulator

A GPU-accelerated, differentiable FMCW radar simulator for generating synthetic radar data from 3D scenes. It combines Mitsuba ray tracing with custom CUDA kernels for scene simulation, signal generation, and downstream radar processing.

This module is derived from [RF-Genesis](https://github.com/Asixa/RF-Genesis).

## Get Started

Python 3.10+ and an NVIDIA GPU are required.
This package depends on the base `witwin` package.

```bash
pip install witwin[radar]
```

## Quick Start

```python
import numpy as np
import torch

from witwin.radar import Radar, RadarConfig
from witwin.radar.sigproc import process_pc, process_rd

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

# Use the recommended GPU backend.
radar = Radar(RadarConfig.from_dict(config), backend="dirichlet", device="cuda")

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
```

## Scene API

Use `Sensor(...)` to define the radar pose, `Scene.set_sensor(...)` to configure the default scene sensor, and `Scene.add_*` methods for scene assembly.

```python
from witwin.core import Material, Structure
from witwin.radar import Scene, Sensor, Simulation

scene = Scene(device="cpu").set_sensor(
    origin=(0.0, 0.0, 0.0),
    target=(0.0, 0.0, -1.0),
    up=(0.0, 1.0, 0.0),
)

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
    rotation={
        "axis": (0.0, 1.0, 0.0),
        "angular_velocity": 32.0,
        "origin": (0.0, 0.0, 0.0),
        "space": "local",
    },
)

result = Simulation.mimo(
    scene,
    config=config,
    backend="dirichlet",
    sampling="triangle",
    motion_sampling="per_chirp",
    device="cpu",
)
```

Available mutating scene methods:

- `Scene.set_sensor(...)`
- `Scene.add_structure(...)`
- `Scene.add_mesh(...)`
- `Scene.add_smpl(...)`
- `Scene.add_structure_motion(...)`
- `Scene.update_structure(...)`
- `Scene.update_structure_motion(...)`
- `Scene.clear_structure_motion(...)`

## Features

- Recommended backend: `dirichlet`
- Ray tracing through Mitsuba with differentiable scene support
- Shared-core geometry and structure primitives
- SMPL body support through `Scene.add_smpl(...)`
- Optional per-structure rigid motion with parent inheritance
- Multi-radar orchestration through `Simulation.mimo_group(...)`
- Torch-native DSP pipeline for range/Doppler processing and point-cloud extraction
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

`amass_pointcloud` and `gen_amass_video` additionally require AMASS BMLmovi data under `data/BMLmovi_full/BMLmovi/`. The rendering examples require `mitsuba` and CUDA; the SMPL examples also require `models/smpl_models/`.
`rgbd_range_doppler` reads `.npy`/`.npz` depth or point-cloud sequences, and can read Azure Kinect `.mkv` files when `pykinect_azure` is installed. It assumes the depth camera view is the radar view by default.

## Installation

Python 3.10+ and an NVIDIA GPU are required.

```bash
pip install witwin[radar]
```

Core dependencies include `torch`, `numpy`, `slangtorch`, `tqdm`, `matplotlib`, and `scipy`. Optional rendering dependencies are `mitsuba` and `drjit`.

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

MIT

## Developer

<a href="http://xingyuchen.me/">
  <img src="https://github.com/Asixa.png" alt="Xingyu Chen" width="48" height="48" style="border-radius:50%;">
</a>

[Xingyu Chen](http://xingyuchen.me/)
