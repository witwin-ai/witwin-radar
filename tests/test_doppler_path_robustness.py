"""A change of tessellation winner must not become a velocity spike."""

import pytest
import torch
from support import multi_endpoint_driver as drv

from witwin.radar.propagation import Kinematics, two_way_duals

pytestmark = pytest.mark.gpu


def test_coplanar_primitive_switch_preserves_delay_field_and_physical_rate():
    frozen = drv.MultiEndpointSpike(
        transmitters=((10, (0.0, 0.0, 0.0)),), receivers=((30, (0.15, 0.0, 0.0)),), sites=((20, (2.0, -0.01, 0.0)),)
    )
    position = torch.tensor([[2.0, 0.01, 0.0]], device="cuda")
    velocity = torch.tensor([[0.0, 1.0, 0.0]], device="cuda")
    fresh = drv.MultiEndpointSpike(
        compiled=frozen.adapter.compiled_scene,
        transmitters=frozen.transmitters,
        receivers=frozen.receivers,
        sites=((20, (2.0, 0.01, 0.0)),),
    )
    outputs = []
    for spike in (frozen, fresh):
        with two_way_duals(sites=Kinematics(position, velocity)) as duals:
            paths, incoming, _ = spike.frame(duals.sites, ad_mode="jvp")
            outputs.append((paths, incoming))
    a, b = (item[0] for item in outputs)
    assert not torch.equal(outputs[0][1].primitive_sequence, outputs[1][1].primitive_sequence)
    assert bool(a.row_valid.all()) and bool(b.row_valid.all())
    torch.testing.assert_close(a.total_delay_s, b.total_delay_s, rtol=0, atol=0)
    torch.testing.assert_close(a.complex_transfer_ref, b.complex_transfer_ref, rtol=0, atol=0)
    torch.testing.assert_close(a.delay_rate, b.delay_rate, rtol=0, atol=1e-15)
    # y motion near boresight contributes only a small projection, even though
    # the winning triangle ID changes discontinuously across the shared edge.
    assert float((77e9 * a.delay_rate).abs().max()) < 4.0
