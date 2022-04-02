"""Launch Blender instances from console.

This application loads a json-serialized message from `jsonargs` argument. The content
of the file has to match the keyword arguments of `btt.BlenderLauncher`. Example
    {
        "scene": "",
        "script": "C:\\dev\\pytorch-blender\\tests\\blender\\launcher.blend.py",
        "num_instances": 2,
        "named_sockets": [
            "DATA",
            "GYM"
        ],
        "background": true,
        "seed": 10
    }
The application than invokes `btt.BlenderLauncher` using these arguments and waits
for the spawned Blender processes to exit. The launch-information is written to
`--out-launch-info` in json format, so that one can connect to the launched intances
from a remote location using LaunchInfo.
"""

from ..launcher import BlenderLauncher
from ..launch_info import LaunchInfo
import json


def main(inargs=None):
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        "Blender Launcher", description=__doc__, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--out-launch-info",
        help="Path to save connection info to.",
        default="launch_info.json",
    )
    parser.add_argument(
        "jsonargs",
        type=str,
        help="JSON Dict of arguments for blendtorch.btt.BlenderLauncher",
    )
    args = parser.parse_args(inargs)

    with open(args.jsonargs, "r") as fp:
        launch_args = json.load(fp)

    # print(json.dumps(launch_args, indent=4))

    with BlenderLauncher(**launch_args) as bl:
        LaunchInfo.save_json(args.out_launch_info, bl.launch_info)
        bl.wait()


if __name__ == "__main__":
    main()
