import pytest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from blendtorch import btt

BLENDDIR = Path(__file__).parent / "blender"


@pytest.mark.background
def test_projection():
    launch_args = dict(
        scene=BLENDDIR / "cam.blend",
        script=BLENDDIR / "cam.blend.py",
        num_instances=1,
        named_sockets=["DATA"],
        background=True,
    )

    ortho_xy_expected = np.array(
        [
            [480.0, 80],
            [480.0, 80],
            [480.0, 400],
            [480.0, 400],
            [160.0, 80],
            [160.0, 80],
            [160.0, 400],
            [160.0, 400],
        ]
    )

    proj_xy_expected = np.array(
        [
            [468.148, 91.851],
            [431.111, 128.888],
            [468.148, 388.148],
            [431.111, 351.111],
            [171.851, 91.851],
            [208.888, 128.888],
            [171.851, 388.148],
            [208.888, 351.111],
        ]
    )

    z_expected = np.array([6.0, 8, 6, 8, 6, 8, 6, 8])

    with btt.BlenderLauncher(**launch_args) as bl:
        addr = bl.launch_info.addresses["DATA"]
        ds = btt.RemoteIterableDataset(addr, max_items=2)
        item = next(iter(ds))
        assert_allclose(item["ortho_xy"], ortho_xy_expected, atol=1e-2)
        assert_allclose(item["ortho_z"], z_expected, atol=1e-2)
        assert_allclose(item["proj_xy"], proj_xy_expected, atol=1e-2)
        assert_allclose(item["proj_z"], z_expected, atol=1e-2)
