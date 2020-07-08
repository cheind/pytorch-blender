import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from contextlib import ExitStack

from blendtorch import btt

def gamma_correct(x):
    '''Return sRGB image.'''
    rgb = x[...,:3].astype(np.float32) / 255
    rgb = np.uint8(255.0 * rgb**(1/2.2))
    return np.concatenate((rgb, x[...,3:4]), axis=-1)

class MyDataset:
    '''A dataset that reads from Blender publishers.'''

    def __init__(self, receiver, image_transform=None, stream_length=64):
        self.receiver = receiver
        self.image_transform = image_transform
        self.stream_length = stream_length

    def __len__(self):
        if self.receiver.is_stream:
            return self.stream_length
        else:
            return len(self.receiver)

    def __getitem__(self, index):        
        # Data is a dictionary of {image, coordinates, process id, frame id} see publisher script
        d = self.receiver.recv(index, timeoutms=10000)
        if self.image_transform:
            d['image'] = self.image_transform(d['image'])
        
        return d['image'], d['xy'], d['btid'], d['frameid']


def iterate(dl):
    DPI=96
    for step, (img, xy, btid, fid) in enumerate(dl):
        print(f'Received batch from Blender processes {btid.numpy()}, frames {fid.numpy()}')
        # Drawing is the slow part ~1.2s, Blender results may be dropped.
        H,W = img.shape[1], img.shape[2]
        fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
        axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5]), fig.add_axes([0.0,0.5,0.5,0.5]), fig.add_axes([0.5,0.5,0.5,0.5])]
        for i in range(img.shape[0]):
            axs[i].imshow(img[i], origin='upper')
            axs[i].scatter(xy[i, :, 0], xy[i, :, 1], s=30)
            axs[i].set_axis_off()
        fig.savefig(f'./tmp/output_{step}.png')
        plt.close(fig)

        
def main():
    # Requires blender to be in path
    # set PATH=c:\Program Files\Blender Foundation\Blender 2.83\;%PATH%
    # set PYTHONPATH=c:\dev\pytorch-blender\pkg_pytorch;c:\dev\pytorch-blender\pkg_blender
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='Blender scene name to run')
    parser.add_argument('--replay', action='store_true', help='Replay from disc instead of launching from Blender')
    args = parser.parse_args()

    with ExitStack() as es:
        if not args.replay:
            bl = es.enter_context(
                btt.BlenderLauncher(
                    num_instances=2, 
                    script=f'scenes/{args.scene}.py', 
                    scene=f'scenes/{args.scene}.blend'
                )
            )
            rec = es.enter_context(
                btt.Recorder('./tmp/record.mpkl')
            )
            receiver = es.enter_context(
                btt.BlenderReceiver(                    
                    recorder=rec
                )
            )
            receiver.connect(bl.addresses)
        else:
            receiver = es.enter_context(btt.FileReceiver('./tmp/record.mpkl'))

        
        ds = MyDataset(receiver, image_transform=gamma_correct)
        # Note, in the following num_workers must be 0
        dl = data.DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)
        # Process data
        iterate(dl)


if __name__ == '__main__':
    main()