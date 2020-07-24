from pathlib import Path
from torch.utils import data

from blendtorch import btt

def main():
    launch_args = dict(
        scene=Path(__file__).parent/'cube.blend',
        script=Path(__file__).parent/'cube.blend.py',
        num_instances=2, 
        named_sockets=['DATA'],
    )

    # Launch Blender
    with btt.BlenderLauncher(**launch_args) as bl:
        # Create remote dataset and limit max length to 16 elements.
        addr = bl.launch_info.addresses['DATA']
        ds = btt.RemoteIterableDataset(addr, max_items=16)
        dl = data.DataLoader(ds, batch_size=4, num_workers=4)
        
        for item in dl:
            # item is a dict with custom content (see cube.blend.py)
            img, xy = item['image'], item['xy']
            print('Received', img.shape, xy.shape)

if __name__ == '__main__':
    main()
