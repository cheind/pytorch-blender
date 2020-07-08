import torch.utils.data as data
import argparse
import time
from contextlib import ExitStack

from blendtorch import btt

from train import MyDataset

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
        
        ds = MyDataset(receiver, stream_length=256)
        dl = data.DataLoader(ds, batch_size=BATCH, num_workers=0, shuffle=False)
        
        t0 = None
        imgshape = None

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