import bpy
import bpy_extras
import zmq
import numpy
import sys
import numpy as np
import os
from PIL import Image

def main():
    ''' Sample script publishing random scene renderings.'''
    argv = sys.argv[1:]
    if '--' in sys.argv:
        sys.argv.index("--")
        argv = sys.argv[sys.argv.index("--") + 1:]

    import argparse
    parser = argparse.ArgumentParser(description='Blender render script.')
    parser.add_argument('bind', help='Bind-to address')
    parser.add_argument('-btid', type=int, help='Identifier for this Blender instance', default=int(os.getpid()))
    args = parser.parse_args(argv)

    # Create a publisher
    ctx = zmq.Context()
    s = ctx.socket(zmq.PUB)
    s.setsockopt(zmq.LINGER, 0)
    s.bind(args.bind)

    # We use our pid as identifier for temporary data
    pid = args.btid

    # Setup scene and random material
    scene = bpy.data.scenes["Scene"]
    scene.render.resolution_percentage = 100
    scene.render.filepath = '//tmp/image_%d.png' % pid
    cube = bpy.data.objects['Cube']
    mat = bpy.data.materials.new('RandomMeshMaterial')
    mat.diffuse_color = np.random.rand(3).tolist()
    cube.active_material = mat

    # Loop until closed from parent process
    while True:
        cube.rotation_euler = np.random.randint(-30,30,3)
        
        bpy.ops.render.render(write_still=True)
        
        width = bpy.context.scene.render.resolution_x 
        height = bpy.context.scene.render.resolution_y

        img = Image.open('./' + scene.render.filepath[2:]).convert('RGB')
        xy = get_pixels(scene, cube)

        # Send dictionary of data to subscribers
        s.send_pyobj({
            'image': np.asarray(img),
            'xy': xy,
            'btid': args.btid
        })

def get_pixels(scene, obj):
    '''Get 2D projection coordinates object's vertex coordinates.'''
    data = obj.data
    mat = obj.matrix_world
    xy = []
    for v in data.vertices:
        p = bpy_extras.object_utils.world_to_camera_view(scene, scene.camera, mat * v.co)
        xy.append(p[:2]) # normalized 2D in W*RS / H*RS
    return np.stack(xy).astype(np.float32)

main()