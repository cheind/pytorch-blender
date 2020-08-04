import pytest
from pathlib import Path

from blendtorch import btt

BLENDDIR = Path(__file__).parent/'blender'

@pytest.mark.background
def test_launcher():
    launch_args = dict(
        scene='',
        script=BLENDDIR/'launcher.blend.py', 
        num_instances=2,        
        named_sockets=['DATA', 'GYM'],
        background=True,
        instance_args=[['--x', '3'],['--x', '4']],
        seed=10
    )
    with btt.BlenderLauncher(**launch_args) as bl:
        addr = bl.launch_info.addresses['DATA']
        ds = btt.RemoteIterableDataset(addr, max_items=2)
        items = [item for item in ds]
        assert len(items) == 2

        first, second = 0, 1
        if items[0]['btid']==1:
            first, second = second, first
        
        assert items[first]['btargs']['btid']==0
        assert items[second]['btargs']['btid']==1
        assert items[first]['btargs']['btseed']==10
        assert items[second]['btargs']['btseed']==11
        assert items[first]['btargs']['btsockets']['DATA'].startswith('tcp://')
        assert items[first]['btargs']['btsockets']['GYM'].startswith('tcp://')
        assert items[second]['btargs']['btsockets']['DATA'].startswith('tcp://')
        assert items[second]['btargs']['btsockets']['GYM'].startswith('tcp://')
        assert items[first]['remainder'] == ['--x', '3']
        assert items[second]['remainder'] == ['--x', '4']
        