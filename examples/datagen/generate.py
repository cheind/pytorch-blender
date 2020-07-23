
from torch.utils.data import DataLoader
from contextlib import ExitStack
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from blendtorch import btt

def gamma_correct(x):
    '''Return sRGB image.'''
    rgb = x[...,:3].astype(np.float32) / 255
    rgb = np.uint8(255.0 * rgb**(1/2.2))
    return np.concatenate((rgb, x[...,3:4]), axis=-1)

def item_transform(item):
    item['image'] = gamma_correct(item['image'])
    return item
        
def iterate(dl):
    DPI=96
    for step, item in enumerate(dl):
        img, xy, btid, fid = item['image'], item['xy'], item['btid'], item['frameid']
        print(f'Received batch from Blender processes {btid.numpy()}, frames {fid.numpy()}')
        H,W = img.shape[1], img.shape[2]
        fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
        axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5]), fig.add_axes([0.0,0.5,0.5,0.5]), fig.add_axes([0.5,0.5,0.5,0.5])]
        for i in range(img.shape[0]):
            axs[i].imshow(img[i], origin='upper')
            axs[i].scatter(xy[i, :, 0], xy[i, :, 1], s=15)
            axs[i].set_axis_off()
            axs[i].set_xlim(0,W-1)
            axs[i].set_ylim(H-1,0)
        fig.savefig(f'./tmp/output_{step}.png')
        plt.close(fig)

BATCH = 4
BLENDER_INSTANCES = 4
WORKER_INSTANCES = 4        

def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='Blender scene name to run')
    parser.add_argument('--replay', action='store_true', help='Replay from disc instead of launching from Blender')
    parser.add_argument('--record', action='store_true', help='Record raw blender data')
    args = parser.parse_args()

    with ExitStack() as es:
        if not args.replay:            
            # Launch Blender instance. Upon exit of this script all Blender instances will be closed.     
            bl = es.enter_context(
                btt.BlenderLauncher(
                    scene=Path(__file__).parent/f'{args.scene}.blend',
                    script=Path(__file__).parent/f'{args.scene}.blend.py',
                    num_instances=BLENDER_INSTANCES, 
                    named_sockets=['DATA'],
                )
            )
            
            # Setup a streaming dataset
            ds = btt.RemoteIterableDataset(
                bl.launch_info.addresses['DATA'], 
                item_transform=item_transform
            )
            # Iterable datasets do not support shuffle
            shuffle = False
            
            # Limit the total number of streamed elements
            ds.stream_length(64)

            # Setup raw recording if desired
            if args.record:
                ds.enable_recording(f'./tmp/record_{args.scene}')
        else:
            # Otherwise we replay from file.
            ds = btt.FileDataset(f'./tmp/record_{args.scene}', item_transform=item_transform)
            shuffle = True
                
        # Setup DataLoader and iterate
        dl = DataLoader(ds, batch_size=BATCH, num_workers=WORKER_INSTANCES, shuffle=shuffle)
        iterate(dl)


if __name__ == '__main__':
    main()