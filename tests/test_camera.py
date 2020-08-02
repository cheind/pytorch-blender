import pytest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from blendtorch import btt

BLENDDIR = Path(__file__).parent/'blender'

@pytest.mark.background
def test_projection():
    launch_args = dict(
        scene=BLENDDIR/'cam.blend',
        script=BLENDDIR/'cam.blend.py', 
        num_instances=1,
        named_sockets=['DATA'],
        background=True,
    )

    ortho_xy_expected = np.array([
        [480.        ,  80],
        [480.        ,  80],
        [480.        , 400],
        [480.        , 400],
        [160.        ,  80],
        [160.        ,  80],
        [160.        , 400],
        [160.        , 400]])

    z_expected = np.array([6., 8, 6, 8, 6, 8, 6, 8])

    with btt.BlenderLauncher(**launch_args) as bl:
        addr = bl.launch_info.addresses['DATA']
        ds = btt.RemoteIterableDataset(addr, max_items=2)
        item = next(iter(ds))
        assert_allclose(item['ortho_xy'], ortho_xy_expected, atol=1e-2)
        assert_allclose(item['ortho_z'], z_expected, atol=1e-2)
        #assert_allclose(item['proj_z'], z_expected, atol=1e-2)