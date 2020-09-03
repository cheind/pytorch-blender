import pytest
from pathlib import Path
from torch.utils.data import DataLoader

from blendtorch import btt

BLENDDIR = Path(__file__).parent/'blender'


@pytest.mark.background
def test_dataset():
    launch_args = dict(
        scene='',
        script=BLENDDIR/'dataset.blend.py', 
        num_instances=1,        
        named_sockets=['DATA'],
        background=True,
    )

    with btt.BlenderLauncher(**launch_args) as bl:
        addr = bl.launch_info.addresses['DATA']

        # Note, https://github.com/pytorch/pytorch/issues/44108
        ds = btt.RemoteIterableDataset(addr, max_items=16)
        dl = DataLoader(ds, batch_size=4, num_workers=4, drop_last=False, shuffle=False)
        
        count = 0
        for item in dl:
            assert item['img'].shape == (4,64,64)
            assert item['frameid'].shape == (4,)
            count += 1


        assert count == 4