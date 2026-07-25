from types import SimpleNamespace

import torch

from witwin.radar.trace import Tracer


def test_render_depth_returns_distance_and_zero_for_misses(monkeypatch):
    tracer = Tracer.__new__(Tracer)
    tracer.resolution = 2
    tracer.radar = SimpleNamespace(device=torch.device("cpu"), position=torch.tensor([1.0, 2.0, 3.0]))
    tracer.scene = SimpleNamespace(compile_renderables=lambda time=None: {"mesh": object()})
    monkeypatch.setattr(tracer, "_prepare_scene", lambda renderables: True)
    packed = torch.tensor(
        [
            [1.0, 2.0, 5.0, 0.2, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 6.0, 3.0, 0.3, 1.0, 0.0, 1.0, 0.0],
            [1.0, 2.0, 3.5, 0.4, 1.0, 1.0, 0.0, 0.0],
        ]
    )
    monkeypatch.setattr(tracer, "_trace_primary_rays", lambda renderables, image_only: packed)

    depth = tracer.render_depth()

    torch.testing.assert_close(depth, torch.tensor([[2.0, 0.0], [5.0, 0.5]]))


def test_render_depth_returns_empty_image_for_empty_scene(monkeypatch):
    tracer = Tracer.__new__(Tracer)
    tracer.resolution = 3
    tracer.radar = SimpleNamespace(device=torch.device("cpu"))
    tracer.scene = SimpleNamespace(compile_renderables=lambda time=None: {})
    monkeypatch.setattr(tracer, "_prepare_scene", lambda renderables: False)

    depth = tracer.render_depth()

    assert depth.shape == (3, 3)
    assert torch.count_nonzero(depth) == 0
