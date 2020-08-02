import bpy
from blendtorch import btb

def main():
    btargs, remainder = btb.parse_blendtorch_args()

    cube = bpy.data.objects['Cube']
    ortho = btb.Camera(bpy.data.objects['CamOrtho'])

    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid, lingerms=5000)
    pub.publish(xy=ortho.object_to_pixel(cube))


main()