import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from contextlib import ExitStack

from blendtorch import btt

class MyDataset:
    '''A dataset that reads from Blender publishers.'''

    def __init__(self, receiver):
        self.receiver = receiver

    def __len__(self):
        if self.receiver.is_stream:
            return 256
        else:
            return len(self.receiver)

    def __getitem__(self, index):        
        d = self.receiver.recv(index, timeoutms=10000)
        return d['image'], d['xy'], d['btid'], d['frameid']

BATCH = 8
INSTANCES = 2
        
def main():
    # Requires blender to be in path
    # set PATH=c:\Program Files\Blender Foundation\Blender 2.83\;%PATH%
    # set PYTHONPATH=c:\dev\pytorch-blender\pkg_pytorch;c:\dev\pytorch-blender\pkg_blender

    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='Blender scene name to run')
    args = parser.parse_args()

    with ExitStack() as es:
        bl = es.enter_context(
            btt.BlenderLauncher(
                num_instances=INSTANCES, 
                script=f'scenes/{args.scene}.py', 
                scene=f'scenes/{args.scene}.blend'
            )
        )
        receiver = es.enter_context(
            btt.BlenderReceiver()
        )
        receiver.connect(bl.addresses)

        
        ds = MyDataset(receiver)
        dl = data.DataLoader(ds, batch_size=BATCH, num_workers=0, shuffle=False)
        
        t0 = None
        imgshape = None
        fids = []

        for item in dl:
            if t0 is None: # 1st is warmup
                t0 = time.time()
                imgshape = item[0].shape

        t1 = time.time()    
        N = len(ds) - BATCH
        B = len(ds)//BATCH - 1
        print(f'Time {(t1-t0)/N:.3f}sec/image, {(t1-t0)/B:.3f}sec/batch, shape {imgshape}')

if __name__ == '__main__':
    main()