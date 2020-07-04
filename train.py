import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse

from blendtorch import torch as bt  

class MyDataset:
    '''A dataset that reads from Blender publishers.'''

    def __init__(self, addresses):
        self.recv = bt.Receiver()
        self.recv.connect(addresses)

    def __len__(self):
        # Virtually anything you'd like to end episodes.
        return 20

    def __getitem__(self, idx):        
        # Data is a dictionary of {image, coordinates, id} see publisher script
        d = self.recv(timeoutms=5000)
        return d['image'], d['xy'], d['btid']
        
def main():
    # Requires blender to be in path
    # set PATH=c:\Program Files\Blender Foundation\Blender 2.83\;%PATH%

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='Blender scene to run')
    args = parser.parse_args()

    with bt.BlenderLauncher(num_instances=2, script=f'scenes/{args.scene}.py', scene=f'scenes/{args.scene}.blend') as bl:
        ds = MyDataset(bl.addresses)

        # Note, in the following num_workers must be 0
        dl = data.DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)

        for step, (img, xy, btid) in enumerate(dl):
            print(f'Received batch from Blender processes {btid}')
            #Drawing is the slow part. It slows down blender rendering to ~8FPS/instance
            fig, axs = plt.subplots(2,2,frameon=False, figsize=(16*2,9*2))
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
            axs = axs.reshape(-1)
            for i in range(img.shape[0]):
                axs[i].imshow(img[i], aspect='auto', origin='upper')
                axs[i].scatter(xy[i, :, 0], xy[i, :, 1], s=100)
                axs[i].set_axis_off()
            fig.savefig(f'./tmp/output_{step}.png')
            plt.close(fig)

if __name__ == '__main__':
    main()