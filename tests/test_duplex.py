import pytest
from pathlib import Path

from blendtorch import btt

BLENDDIR = Path(__file__).parent/'blender'

@pytest.mark.background
def test_duplex():
    launch_args = dict(
        scene='',
        script=BLENDDIR/'duplex.blend.py', 
        num_instances=1,        
        named_sockets=['CTRL'],
        background=True,
    )

    with btt.BlenderLauncher(**launch_args) as bl:

        addr = bl.launch_info.addresses['CTRL']
        duplex = btt.DuplexChannel(addr[0], lingerms=5000)
        duplex.send(dict(msg='hello'))
        msgs = duplex.recv(timeoutms=5000)
        assert len(msgs) == 2
        assert msgs[0]['got'] == [{'msg': 'hello'}]
        assert msgs[1] == 'end'