import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse

from blendtorch import torch as bt  

class MyDataset:
    '''A dataset that reads from Blender publishers.'''

    def __init__(self, addresses, transforms=None):
        self.recv = bt.Receiver()
        self.recv.connect(addresses)
        self.transforms = transforms

    def __len__(self):
        # Virtually anything you'd like to end episodes.
        return 20

    def __getitem__(self, idx):        
        # Data is a dictionary of {image, coordinates, id} see publisher script
        d = self.recv(timeoutms=5000)

        return d['image'], d['btid']
        
        # x, coords = d['image'], d['xy']        
        # h,w = x.shape[0], x.shape[1]
        # coords[...,1] = 1. - coords[...,1] # Blender uses origin bottom-left.        
        # coords *= np.array([w,h])[None, :]

        # if self.transforms:
        #     x = self.transforms(x)
        # return x, coords, d['btid']

def main():
    # Requires blender to be in path
    # set PATH=c:\Program Files\Blender Foundation\Blender 2.83\;%PATH%

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='Blender scene to run')
    args = parser.parse_args()

    with bt.BlenderLauncher(num_instances=2, script=f'scenes/{args.scene}.py', scene='scenes/{args.scene}.blend') as bl:
        ds = MyDataset(bl.addresses)

        # Note, in the following num_workers must be 0
        dl = data.DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)

        import time
        t = None
        idx = 0
        for imgs, ids in dl:
            if t is None:
                t = time.time()
            print(ids)

        print(time.time() - t)

        # for idx, (x, coords, ids) in enumerate(dl):
        #     print(f'Received from Blender processes {ids.cpu().numpy()}')

            # Drawing is the slow part ...
            # fig, axs = plt.subplots(2,2,frameon=False, figsize=(16*2,9*2))
            # fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
            # axs = axs.reshape(-1)
            # for i in range(x.shape[0]):
            #     axs[i].imshow(x[i], aspect='auto', origin='upper')
            #     axs[i].scatter(coords[i][:, 0], coords[i][:, 1], s=100)
            #     axs[i].set_axis_off()
            # fig.savefig(f'./tmp/output_{idx}.png')
            # plt.close(fig)

if __name__ == '__main__':
    main()