import bpy
from blendtorch import btb


def main():
    btargs, remainder = btb.parse_blendtorch_args()

    cam = btb.Camera()
    render = btb.CompositeRenderer(
        [
            btb.CompositeSelection('color', 'File Output', 'Color', 'RGB'),
            btb.CompositeSelection('depth', 'File Output', 'Depth', 'V'),
        ],
        btid=btargs.btid,
        camera=cam,
    )
    data = render.render()
    pub = btb.DataPublisher(
        btargs.btsockets['DATA'],
        btargs.btid,
        lingerms=5000
    )
    pub.publish(**data)


main()
