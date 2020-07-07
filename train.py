import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse

from blendtorch import btt

def gamma_correct(x):
    '''Return sRGB image.'''
    rgb = x[...,:3].astype(np.float32) / 255
    rgb = np.uint8(255.0 * rgb**(1/2.2))
    return np.concatenate((rgb, x[...,3:4]), axis=-1)

class MyDataset:
    '''A dataset that reads from Blender publishers.'''

    def __init__(self, addresses):
        self.recv = btt.Subscriber()
        self.recv.connect(addresses)

    def __len__(self):
        # Virtually anything you'd like to end episodes.
        return 64

    def __getitem__(self, idx):        
        # Data is a dictionary of {image, coordinates, id} see publisher script
        d = self.recv(timeoutms=10000)
        return gamma_correct(d['image']), d['xy'], d['btid'], d['frameid']
        
def main():
    # Requires blender to be in path
    # set PATH=c:\Program Files\Blender Foundation\Blender 2.83\;%PATH%
    # set PYTHONPATH=c:\dev\pytorch-blender\pkg_pytorch;c:\dev\pytorch-blender\pkg_blender
    logging.basicConfig(level=logging.INFO)
    DPI=96

    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='Blender scene to run')
    args = parser.parse_args()

    with btt.BlenderLauncher(num_instances=2, script=f'scenes/{args.scene}.py', scene=f'scenes/{args.scene}.blend') as bl:
        ds = MyDataset(bl.addresses)

        # Note, in the following num_workers must be 0
        dl = data.DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)

        for step, (img, xy, btid, fid) in enumerate(dl):
            print(f'Received batch from Blender processes {btid.numpy()}, frames {fid.numpy()}')
            # Drawing is the slow part ~1.2s, Blender results may be dropped.
            H,W = img.shape[1], img.shape[2]
            fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=96)
            axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5]), fig.add_axes([0.0,0.5,0.5,0.5]), fig.add_axes([0.5,0.5,0.5,0.5])]
            for i in range(img.shape[0]):
                axs[i].imshow(img[i], origin='upper')
                axs[i].scatter(xy[i, :, 0], xy[i, :, 1], s=100)
                axs[i].set_axis_off()
            fig.savefig(f'./tmp/output_{step}.png')
            plt.close(fig)

if __name__ == '__main__':
    main()