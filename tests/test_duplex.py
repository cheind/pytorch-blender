import pytest
from pathlib import Path

from blendtorch import btt

BLENDDIR = Path(__file__).parent / "blender"


#@pytest.mark.background
def test_duplex():
    launch_args = dict(
        scene="",
        script=BLENDDIR / "duplex.blend.py",
        num_instances=2,
        named_sockets=["CTRL"],
        background=True,
    )

    with btt.BlenderLauncher(**launch_args) as bl:

        addresses = bl.launch_info.addresses["CTRL"]
        duplex = [btt.DuplexChannel(addr, lingerms=5000) for addr in addresses]
        mids = [d.send(msg=f"hello {i}") for i, d in enumerate(duplex)]

        def rcv_twice(d):
            return [
                d.recv(timeoutms=5000),
                d.recv(timeoutms=5000),
            ]

        msgs = [rcv_twice(d) for d in duplex]

        assert len(msgs) == 2
        assert len(msgs[0]) == 2

        assert msgs[0][0]["echo"]["msg"] == "hello 0"
        assert msgs[0][0]["echo"]["btid"] is None
        assert msgs[0][0]["echo"]["btmid"] == mids[0]
        assert msgs[0][0]["btid"] == 0
        assert msgs[0][1]["msg"] == "end"
        assert msgs[0][1]["btid"] == 0

        assert msgs[1][0]["echo"]["msg"] == "hello 1"
        assert msgs[1][0]["echo"]["btid"] is None
        assert msgs[1][0]["echo"]["btmid"] == mids[1]
        assert msgs[1][0]["btid"] == 1
        assert msgs[1][1]["msg"] == "end"
        assert msgs[1][1]["btid"] == 1
