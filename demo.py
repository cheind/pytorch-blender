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

    def __init__(self, inchannel, image_transform=None, stream_length=256):
        self.inchannel = inchannel
        self.image_transform = image_transform
        self.stream_length = stream_length

    def __len__(self):
        if self.inchannel.is_stream:
            return self.stream_length
        else:
            return len(self.inchannel)

    def __getitem__(self, index):        
        # Data is a dictionary of {image, coordinates, process id, frame id} see publisher script
        d = self.inchannel.recv(index, timeoutms=10000)
        if self.image_transform:
            d['image'] = self.image_transform(d['image'])
        
        return d['image'], d['xy'], d['btid'], d['frameid']


def iterate(dl):
    DPI=96
    for step, (img, xy, btid, fid) in enumerate(dl):
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
        
def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='Blender scene name to run')
    parser.add_argument('--replay', action='store_true', help='Replay from disc instead of launching from Blender')
    parser.add_argument('--record', action='store_true', help='Record raw blender data')
    args = parser.parse_args()

    BATCH = 4
    BLENDER_INSTANCES = 4
    WORKER_INSTANCES = 1

    with ExitStack() as es:
        if not args.replay:
            # Initiate Blender instance     
            bl = es.enter_context(
                btt.BlenderLauncher(
                    scene=f'scenes/{args.scene}.blend',
                    script=f'scenes/{args.scene}.py', 
                    num_instances=BLENDER_INSTANCES, 
                    named_sockets=['DATA'],
                )
            )
            # Add recording if needed (requires num_workers=0)
            rec = None
            if args.record:
                rec = es.enter_context(
                    btt.Recorder('./tmp/record.mpkl')
                )
                WORKER_INSTANCES = 0            
            # Add receiver
            inchan = btt.BlenderInputChannel(                    
                recorder=rec,
                addresses=bl.launch_info.addresses['DATA']
            )
        else:
            inchan = btt.FileInputChannel('./tmp/record.mpkl')
        
        ds = MyDataset(inchan, image_transform=gamma_correct)
        # Note, in the following num_workers must be 0
        dl = data.DataLoader(ds, batch_size=BATCH, num_workers=WORKER_INSTANCES, shuffle=False)
        # Process data
        iterate(dl)


if __name__ == '__main__':
    main()