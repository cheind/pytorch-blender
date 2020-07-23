from pathlib import Path
from torch.utils import data
from blendtorch import btt

BATCH = 4
BLENDER = 2
WORKER = 4  

def main():
    launch_args = dict(
        scene=Path(__file__).parent/'cube.blend',
        script=Path(__file__).parent/'cube.blend.py',
        num_instances=BLENDER, 
        named_sockets=['DATA'],
    )

    with btt.BlenderLauncher(**launch_args) as bl:
        addr = bl.launch_info.addresses['DATA']
        
        # Create remote dataset
        ds = btt.RemoteIterableDataset(addr)

        # Limit the total number of streamed elements
        ds.stream_length(16)

        # Setup batching
        dl = data.DataLoader(ds, batch_size=BATCH, num_workers=WORKER)
        
        # Loop
        for item in dl:
            img, xy = item['image'], item['xy']
            print(img.shape, xy.shape)

if __name__ == '__main__':
    main()
