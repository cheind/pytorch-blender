## Supervised Training Data Generation

This directory showcases synthetic data generation using **blendtorch** for supervised machine learning. In particular, several blender processes render randomized scene configurations and stream images as well as annotations into a PyTorch dataset used in training neural networks. 

The figure below visualizes a single image/label batch received by PyTorch from four parallel Blender instances. Each Blender process repeatedly performs motion simulations of randomized cubes.

<p align="center">
<img src="etc/result_physics.png" width="500">
</p>

### Run

To recreate these results run [generate.py](./generate.py) using the [falling_cubes](./) scene as follows
```
python generate.py falling_cubes
```
which will generate output images in `./tmp/output_##.png`. 

### Record and replay

See [generate.py](./generate.py) for usage of recording and replay features.
Saving images is only done for demonstration purposes. **blendtorch** does not require intermediate disk storage to run.

### Minimal example
The following snippets show the minimal necessary code to use **blendtorch** for connecting PyTorch datasets to Blender renderings/ annotations. To run the example, invoke
```
python minimal.py
```

#### PyTorch part
On the PyTorch side, [minimal.py](./minimal.py) the following steps:

```python
from pathlib import Path
from torch.utils import data

from blendtorch import btt

def main():
    launch_args = dict(
        scene=Path(__file__).parent/'cube.blend',
        script=Path(__file__).parent/'cube.blend.py',
        num_instances=2, 
        named_sockets=['DATA'],
    )

    # Launch Blender
    with btt.BlenderLauncher(**launch_args) as bl:
        # Create remote dataset and limit max length to 16 elements.
        addr = bl.launch_info.addresses['DATA']
        ds = btt.RemoteIterableDataset(addr, max_items=16)
        dl = data.DataLoader(ds, batch_size=4, num_workers=4)
        
        for item in dl:
            # item is a dict with custom content (see cube.blend.py)
            img, xy = item['image'], item['xy']
            print('Received', img.shape, xy.shape)

if __name__ == '__main__':
    main()
```
#### Blender part
When [minimal.py](./minimal.py) launches Blender, each instance will be running 
scene [cube.blend](./cube.blend) and script [cube.blend.py](./cube.blend.py) containing:

```python
import bpy
import numpy as np

from blendtorch import btb

def main():
    # Parse script arguments passed via blendtorch
    btargs, remainder = btb.parse_blendtorch_args()

    cam = bpy.context.scene.camera
    cube = bpy.data.objects["Cube"]

    def pre_frame():
        # Randomize cube rotation
        cube.rotation_euler = np.random.uniform(0,np.pi,size=3)  
        
    def post_frame(off, pub, anim, cam):
        # Called every after Blender finished processing a frame.
        pub.publish(
            image=off.render(), 
            xy=cam.object_to_pixel(cube), 
            frameid=anim.frameid
        )

    # Data source
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)

    # Setup default image rendering
    cam = btb.Camera()
    off = btb.OffScreenRenderer(camera=cam, mode='rgb')
    off.set_render_style(shading='RENDERED', overlays=False)

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame)
    anim.post_frame.add(post_frame, off, pub, anim, cam)    
    anim.play(frame_range=(0,100), num_episodes=-1)

main()
```

### Launching and connecting to remote instances
Often you will find it convenient to launch Blender instances on a machine *A* while model training is supposed to happen on machine *B*. To facilitate this use case, **blendtorch** comes with a set of supporting tools.

From here on we assume that *A* has `pkg_pytorch` and `pkg_blender` installed, while *B* has at least `pkg_pytorch` installed.

On *A* run
```
blendtorch-launch launch.json
```
where `launch.json` contains a dictionary of keyword arguments for `btt.BlenderLauncher`. For example
```json
{
    "scene": "",
    "script": "/tests/blender/launcher.blend.py",
    "num_instances": 2,
    "named_sockets": [
        "DATA",
        "GYM"
    ],
    "background": true,
    "seed": 10,
    "bind_addr": "primaryip"
}
```
Upon launch, `blendtorch-launch` writes connection information to `launch_info.json`. For example
```json
{
    "addresses": {
        "DATA": [
            "tcp://192.168.20.148:11000",
            "tcp://192.168.20.148:11001"
        ],
        "GYM": [
            "tcp://192.168.20.148:11002",
            "tcp://192.168.20.148:11003"
        ]
    },
}
```
Notice, `primaryip` was automatically resolved into an IP addresses with default route configured. Now, to connect from *B*, ensure the machine has access to `launch_info.json` and connect as follows
```python
import blendtorch.btt as btt
launch_info = btt.LaunchInfo.load_json('launch_info.json')
ds = btt.RemoteIterableDataset(launch_info.addresses['DATA'], max_items=2)
item = next(iter(ds))
print(item.keys())
#...
```

### Architecture

**blendtorch** is composed of two distinct sub-packages: `bendtorch.btt` (in [pkg_pytorch](./pkg_pytorch)) and `blendtorch.btb` (in [pkg_blender](./pkg_blender)), providing the PyTorch and Blender views on **blendtorch**. 

In data streaming, we are interested in sending supervised image data from multiple Blender processes to a Python process running model training. This process is depicted below.

<p align="center">
<img src="./etc/blendtorch_datagen.svg" width="500">
</p>

Typically a Python script, e.g `train.py`, launches and maintains one or more Blender instances using `blendtorch.btt.BlenderLauncher`. Each Blender instance will be instructed to run particular scene and script, e.g `blend.py`. Next, `train.py` creates a `RemoteIterableDataset` to listen for incoming network messages from Blender instances. We use a `PUSH/PULL` pipeline pattern that supports fair queuing and will stall Blender instances when `train.py` is too slow to process all messages. 

Each Blender instance, running `blend.py`, meanwhile creates a `blendtorch.btb.DataPublisher` to send outward messages. The addresses are taken from command-line arguments and are automatically provided by `blendtorch.btt.BlenderLauncher`. Next, `blend.py` registers the necessary animation hooks and usually creates one or more `blendtorch.btb.OffScreenRenderer` to capture offscreen images. Usually at `pre_frame` callbacks the scene is randomized and during `post_frame` the resulting frame is rendered and sent via output channel alongside with any (pickle-able) meta information desired.

#### Distributed computing
**blendtorch** supports two kinds of data parallism: Blender instances and PyTorch workers. We use a [PUSH/PULL pattern](https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pushpull.html) that allows us to fan out from multiple Blender instances and distribute the workload to any number of PyTorch workers. 

It is guaranteed that only one PyTorch worker receives a particular message, no message is lost, but the order in which it is received is not guaranteed. When PyTorch is too slow to process all messages in time, the Blender instances will eventually block until new slosts are available. When the number of PyTorch workers is one (i.e `num_workers=0` in DataLoader) then all messages will be received in the order they have been generated. 

Every PyTorch worker interleaves  messages from all connected Blender instances in a fair manner. You may use the `btid` message field to determine which Blender instance sent which message.



