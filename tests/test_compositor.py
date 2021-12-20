import pytest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from blendtorch import btt

BLENDDIR = Path(__file__).parent / "blender"


# @pytest.mark.background
# Seems to not run on travis
def test_projection():
    launch_args = dict(
        scene=BLENDDIR / "compositor.blend",
        script=BLENDDIR / "compositor.blend.py",
        num_instances=1,
        named_sockets=["DATA"],
        background=True,
    )

    expected_color = np.full((200, 320, 3), (0, 1, 0), dtype=np.float32)
    expected_depth = np.full((200, 320, 1), 2.0, dtype=np.float32)

    with btt.BlenderLauncher(**launch_args) as bl:
        addr = bl.launch_info.addresses["DATA"]
        ds = btt.RemoteIterableDataset(addr, max_items=1)
        item = next(iter(ds))
        assert_allclose(item["color"], expected_color)
        assert_allclose(item["depth"], expected_depth)
