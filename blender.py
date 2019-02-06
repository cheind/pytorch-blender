import bpy
import zmq
import numpy
import sys
import numpy as np

def main():
    argv = sys.argv[1:]
    if '--' in sys.argv:
        sys.argv.index("--")
        argv = sys.argv[sys.argv.index("--") + 1:]

    import argparse
    parser = argparse.ArgumentParser(description='Blender render script.')
    parser.add_argument('bind', help='Bind-to address')
    args = parser.parse_args(argv)

    ctx = zmq.Context()
    s = ctx.socket(zmq.PUB)
    s.setsockopt(zmq.LINGER, 0)
    s.bind(args.bind)

    scene = bpy.data.scenes["Scene"]
    scene.render.resolution_percentage = 100

    cube = bpy.data.objects['Cube']
    cube.color = np.random.rand(3).tolist() + [1]

    while True:
        cube.rotation_euler = np.random.randint(-30,30,3)

        bpy.ops.render.render()
        pixels = np.array(bpy.data.images['Viewer Node'].pixels)
        print(len(pixels))

        width = bpy.context.scene.render.resolution_x 
        height = bpy.context.scene.render.resolution_y

        image = pixels.reshape((height, width, 4))
        s.send_pyobj(image)

main()