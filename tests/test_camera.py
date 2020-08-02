import pytest
from pathlib import Path

from blendtorch import btt

BLENDDIR = Path(__file__).parent/'blender'

@pytest.mark.background
def test_cam_projection():
    launch_args = dict(
        scene=BLENDDIR/'cam.blend',
        script=BLENDDIR/'cam.blend.py', 
        num_instances=1,
        named_sockets=['DATA'],
        background=True,
    )
    with btt.BlenderLauncher(**launch_args) as bl:
        addr = bl.launch_info.addresses['DATA']
        ds = btt.RemoteIterableDataset(addr, max_items=2)

        item = next(iter(ds))
        print(item)