# infra

## Breaking changes

- The `channel` extra now requires `witwin-channel>=0.6.0,<0.7` (consumer contract 7). Migration: upgrade
  Channel with `pip install -U "witwin-radar[channel]"`; Radar 0.5.0 does not run on Channel 0.5.
- The `channel` extra installs nothing on Python 3.14, because Channel 0.6 publishes wheels for CPython
  3.10-3.13 only. Radar itself still installs on 3.10-3.14; scene-driven simulation needs Python 3.10-3.13.
- A forward-mode tangent on mesh vertices is no longer available: Channel 0.6 refuses to compile a mesh whose
  vertices carry a forward-AD dual (`NotImplementedError` naming `aminmax`). Migration: take vertex derivatives in
  reverse mode (`grad="vjp"`), which is unchanged.

## Changed

- Reflection rows follow Channel 0.6's path-validity rules (Channel ADR-059, ADR-068): a reflection whose
  specular point lies past the edge of a face by more than a few float roundings is no longer reported (Channel
  0.5 accepted points up to about 1e-4 of the face's altitude past the edge), and a reflection with a segment
  shorter than 1e-5 of the scene diagonal is no longer reported, so a scatter site within that distance of a
  wall loses that wall's reflection.
- A scene edited in place after it was compiled is refused by Channel at the next replay ("the compiled scene
  is stale") instead of answering from the old geometry. Radar's own motion-event tick still detects the edit
  and recompiles when a motion-event period is declared.
- Channel 0.6 picks its ray-tracing backend when it compiles a scene and falls back to a pure-CUDA tracer
  where OptiX is unavailable; `witwin-channel-doctor` reports which one a machine gets.

## Fixed

- The Linux native library links with `--as-needed`, so it no longer records an unused dependency on
  `libtorch_python.so`.
- `pytest tests` collects and runs without a CUDA device.
