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
        ds = btt.RemoteIterableDataset(addr, max_items=16)
        dl = data.DataLoader(ds, batch_size=4, num_workers=0)

        for item in dl:
            # item is a dict with custom content (see cube.blend.py)
            normals = item['normals']
            depth = item['depth']
            print('Received', normals.shape, depth.shape,
                  depth.dtype, np.ptp(depth))

            fig, axs = plt.subplots(2,2)
            axs = np.asarray(axs).reshape(-1)
            for i in range(4):
                axs[i].imshow(depth[i, :, :, 0], vmin=1, vmax=2.5)
            fig, axs = plt.subplots(2,2)
            axs = np.asarray(axs).reshape(-1)
            for i in range(4):
                axs[i].imshow(normals[i, :, :])
            plt.show()


if __name__ == '__main__':
    main()
