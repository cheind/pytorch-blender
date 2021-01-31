import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils import data


import blendtorch.btt as btt


def main():
    # Define how we want to launch Blender
    launch_args = dict(
        scene=Path(__file__).parent/'normals_depth.blend',
        script=Path(__file__).parent/'normals_depth.blend.py',
        num_instances=1,
        named_sockets=['DATA'],
    )

    # Launch Blender
    with btt.BlenderLauncher(**launch_args) as bl:
        # Create remote dataset and limit max length to 16 elements.
        addr = bl.launch_info.addresses['DATA']
        ds = btt.RemoteIterableDataset(addr, max_items=4)
        dl = data.DataLoader(ds, batch_size=4, num_workers=0)

        for item in dl:
            # item is a dict with custom content (see cube.blend.py)
            normals = item['normals']
            depth = item['depth']
            print('Received', normals.shape, depth.shape,
                  depth.dtype, np.ptp(depth))

            plt.figure()
            plt.imshow(depth[0, :, :, 0], vmin=0, vmax=4.)
            plt.figure()
            plt.imshow(normals[0, :, :])
            plt.show()
        # Will get here after 16/BATCH=4 iterations.


if __name__ == '__main__':
    main()
