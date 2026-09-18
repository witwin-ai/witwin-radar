"""A change of tessellation winner must not change the delay field."""

import pytest
import torch
from support import multi_endpoint_driver as drv

pytestmark = pytest.mark.gpu


def test_coplanar_primitive_switch_preserves_the_delay_field():
    frozen = drv.MultiEndpointSpike(
        transmitters=((10, (0.0, 0.0, 0.0)),), receivers=((30, (0.15, 0.0, 0.0)),), sites=((20, (2.0, -0.01, 0.0)),)
    )
    position = torch.tensor([[2.0, 0.01, 0.0]], device="cuda")
    fresh = drv.MultiEndpointSpike(
        compiled=frozen.adapter.compiled_scene,
        transmitters=frozen.transmitters,
        receivers=frozen.receivers,
        sites=((20, (2.0, 0.01, 0.0)),),
    )
    outputs = [spike.frame(position)[:2] for spike in (frozen, fresh)]
    a, b = (item[0] for item in outputs)
    # The winning triangle ID changes discontinuously across the shared edge;
    # the round trip through the shared plane does not.
    assert not torch.equal(outputs[0][1].primitive_sequence, outputs[1][1].primitive_sequence)
    assert bool(a.row_valid.all()) and bool(b.row_valid.all())
    torch.testing.assert_close(a.total_delay_s, b.total_delay_s, rtol=0, atol=0)
    torch.testing.assert_close(a.complex_transfer_ref, b.complex_transfer_ref, rtol=0, atol=0)
