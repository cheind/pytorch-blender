## Supervised Training Data Generation

This directory showcases synthetic data generation using **blendtorch** for supervised machine learning. In particular, several blender processes render randomized scene configurations and stream images as well as annotations into a PyTorch dataset used in training neural networks.

Shown below is a result visualization from 4 Blender instances running physics-enabled falling cubes scene.

<div style="text-align: center">
    <img src="/etc/result_physics.png" width="500">
</div>

To recreate these results run [generate.py](./generate.py) using the [falling_cubes](./) as follows
```
python generate.py falling_cubes
```
which will generate output images in `./tmp/output_##.png`. Note that saving images these images is only done for demonstration purposes. **blendtorch** does not require intermediate disk storage to run.

## Minimal sample
### PyTorch
The following [minimal.py](./minimal.py) sample shows the steps necessary to launch and receive from multiple Blender processes in PyTorch.

```python
from pathlib import Path
from torch.utils import data
from blendtorch import btt

BATCH = 4
BLENDER = 2
WORKER = 4  

def main():
    launch_args = dict(
        scene=Path(__file__).parent/'cube.blend',
        script=Path(__file__).parent/'cube.blend.py',
        num_instances=BLENDER, 
        named_sockets=['DATA'],
    )

    with btt.BlenderLauncher(**launch_args) as bl:
        addr = bl.launch_info.addresses['DATA']
        
        # Create remote dataset
        ds = btt.RemoteIterableDataset(addr)

        # Limit the total number of streamed elements
        ds.stream_length(16)

        # Setup batching
        dl = data.DataLoader(ds, batch_size=BATCH, num_workers=WORKER)
        
        # Loop
        for item in dl:
            img, xy = item['image'], item['xy']
            print(img.shape, xy.shape)

if __name__ == '__main__':
    main()
```
### Blender
The above script launches Blender instances running scene [cube.blend](./cube.blend) and script [cube.blend.py](./cube.blend.py) which contains the following code necessary for streaming

```python
import bpy
import numpy as np
from blendtorch import btb

def main():
    args, remainder = btb.parse_blendtorch_args()

    cam = bpy.context.scene.camera
    obj = bpy.data.objects["Cube"]
    mat = bpy.data.materials.new(name='cube_random')
    obj.data.materials.append(mat)
    obj.active_material = mat
        
    def pre_frame():
        # Called every time before a frame is processed.
        obj.rotation_euler = np.random.uniform(0,np.pi,size=3)  
        mat.diffuse_color = np.concatenate((np.random.random(size=3), [1.]))
        
    def post_frame(off, pub, anim):
        # Called every after Blender finished processing a frame.
        pub.publish(
            image=off.render(), 
            xy=btb.camera.project_points(obj, camera=cam),
            frameid=anim.frameid
        )

    # Our output channel
    pub = btb.BlenderOutputChannel(args.btsockets['DATA'], args.btid)

    # Setup image rendering
    off = btb.OffScreenRenderer(mode='rgb')
    off.view_matrix = btb.camera.view_matrix()
    off.proj_matrix = btb.camera.projection_matrix()
    off.set_render_style(shading='RENDERED', overlays=False)

    # Setup the animation and run endlessly
    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame)
    anim.post_frame.add(post_frame, off, pub, anim)    
    anim.play(frame_range=(0,100), num_episodes=-1)

main()
```

