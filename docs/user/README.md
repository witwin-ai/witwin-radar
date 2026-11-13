# WiTwin Radar user guides

Start with the [quickstart](quickstart.md), then read [conventions and units](conventions.md) before writing your
own scenes: the radar's default pose is Y-up while Channel and Core scenes are Z-up, and positive velocity means
closing. Every Python block in these guides runs as written on a CUDA machine with Channel installed
(`python tools/run_doc_snippets.py docs/user/<guide>.md`).

| Guide | What it covers |
|---|---|
| [Quickstart](quickstart.md) | Install, the first Range-Doppler map, a wall and multipath, reading a `Result`, the four verbs |
| [Conventions and units](conventions.md) | World and radar frames, the default pose against Channel's Z-up world, units, velocity sign, phasors, `Mesh(recenter=False)` |
| [Choosing `Motion`](motion.md) | `auto`, `adaptive`, `adc`, `chirp`; discovery cadences, path events, ego-motion, what each costs |
| [Targets](targets.md) | `PointTargets`, `Aspect`, `ScatteringMatrix`, `RcsTable`, `Fluctuation`, `StructureTargets`, `MeshTargets`, materials, roughness, SMPL bodies |
| [Multipath and environment](multipath.md) | Reflections, transmission, diffraction, environment and coupling rows, completeness, scenes from Channel's importers and presets |
| [Waveforms and MIMO](waveforms.md) | FMCW and chirp sequences, DDM/BPM, OFDM (CFR and time domain), pulsed and coded pulses |
| [Receive chain](receive-chain.md) | Patterns, polarization, coupling, leakage, LNA, compression, noise, DC offset, IQ, AGC, ADC |
| [Multi-radar networks](networks.md) | `Network`, bistatic links, mutual interference, emitters |
| [Processing cookbook](processing.md) | Range-Doppler, CFAR, angles, point clouds, clustering, tracking, micro-Doppler, imaging, calibration, recorded cubes |
| [Gradients and inverse design](gradients.md) | `ad_mode`, the differentiable leaves, an inverse-design loop, what is refused |
| [Datasets and tools](datasets-and-tools.md) | The doctor, `capabilities()`, save/load, configuration files, TI import, DCA1000, labels, the dataset writer |
| [Troubleshooting](troubleshooting.md) | Common errors and what to do about them |

Reference material: [`docs/pipeline_guide.md`](../pipeline_guide.md) (the full pipeline contract),
[`docs/configuration.md`](../configuration.md) (the flat configuration format),
[`docs/migration-0.5.md`](../migration-0.5.md) (upgrading from 0.4), [`CHANGELOG.md`](../../CHANGELOG.md) and
[`FEATURE_LIST.md`](../../FEATURE_LIST.md).
