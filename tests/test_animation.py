import pytest
from pathlib import Path
from blendtorch import btt

BLENDDIR = Path(__file__).parent / "blender"

EXPECTED = [
    "pre_play",
    1,
    "pre_animation",
    1,
    "pre_frame",
    1,
    "post_frame",
    1,
    "pre_frame",
    2,
    "post_frame",
    2,
    "pre_frame",
    3,
    "post_frame",
    3,
    "post_animation",
    3,
    "pre_animation",
    1,
    "pre_frame",
    1,
    "post_frame",
    1,
    "pre_frame",
    2,
    "post_frame",
    2,
    "pre_frame",
    3,
    "post_frame",
    3,
    "post_animation",
    3,
    "post_play",
    3,
]


def _capture_anim_callback_sequence(background):
    launch_args = dict(
        scene="",
        script=BLENDDIR / "anim.blend.py",
        named_sockets=["DATA"],
        background=background,
    )

    with btt.BlenderLauncher(**launch_args) as bl:
        addr = bl.launch_info.addresses["DATA"]
        ds = btt.RemoteIterableDataset(addr, max_items=1, timeoutms=10000)
        try:
            item = next(iter(ds))
            assert item["seq"] == EXPECTED
        except Exception:
            print("err")


@pytest.mark.background
def test_anim_callback_sequence():
    _capture_anim_callback_sequence(background=True)


def test_anim_callback_sequence_ui():
    _capture_anim_callback_sequence(background=False)
