import time
import argparse
from pathlib import Path
import torch.utils.data as data
from blendtorch import btt

BATCH = 8
INSTANCES = 4
WORKER_INSTANCES = 4
NUM_ITEMS = 512
EXAMPLES_DIR = Path(__file__).parent/'..'/'examples'/'datagen'
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', help='Blender scene name to run', default='cube')
    args = parser.parse_args()

    launch_args = dict(
        scene=EXAMPLES_DIR/f'{args.scene}.blend',
        script=EXAMPLES_DIR/f'{args.scene}.blend.py', 
        num_instances=INSTANCES,
        named_sockets=['DATA']
    )

    with btt.BlenderLauncher(**launch_args) as bl:                        
        ds = btt.RemoteIterableDataset(bl.launch_info.addresses['DATA'])
        ds.stream_length(NUM_ITEMS)
        dl = data.DataLoader(ds, batch_size=BATCH, num_workers=WORKER_INSTANCES, shuffle=False)

        # Wait to avoid timing startup times of Blender
        time.sleep(5)
        
        t0 = None
        imgshape = None
        
        n = 0
        for item in dl:
            if t0 is None: # 1st is warmup
                t0 = time.time()
                imgshape = item['image'].shape
            n += len(item['image'])
        assert n == NUM_ITEMS

        t1 = time.time()    
        N = NUM_ITEMS - BATCH
        B = NUM_ITEMS//BATCH - 1
        print(f'Time {(t1-t0)/N:.3f}sec/image, {(t1-t0)/B:.3f}sec/batch, shape {imgshape}')

if __name__ == '__main__':
    main()