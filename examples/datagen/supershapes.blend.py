import numpy as np
import bpy
import logging
import supershape as sshape
import blendtorch.btb as btb

SCN = bpy.context.scene
SCOLL = SCN.collection.children['Supershapes']

def generate_supershapes(n=10, shape=(50,50)):
    sshapes = [sshape.make_bpy_mesh(shape, name='sshape', coll=SCOLL) for i in range(n)]

    params = np.random.uniform(
        low =[0.00,1,1,1.0,0.0, 0.0],
        high=[20.00,1,1,40,10.0,10.0],
        size=(n,2,6)
    )
    scale = np.random.uniform(0.1, 0.6, size=(n,3))
    for i in range(n):
        obj = sshapes[i]
        x,y,z = sshape.supercoords(params[i], shape=shape)
        sshape.update_bpy_mesh(x*scale[i,0], y*scale[i,1], z*scale[i,2],obj)
        obj.location = np.random.uniform(low=[-2, -2, 3],high=[2,2,6],size=(3))
        obj.rotation_euler = np.random.uniform(low=-np.pi, high=np.pi,size=(3))
        SCN.rigidbody_world.collection.objects.link(obj)

    return sshapes

def remove_supershapes(sshapes):
    for s in sshapes:
        s.data.materials.clear()
        SCOLL.objects.unlink(s)
        SCN.rigidbody_world.collection.objects.unlink(s)    
        bpy.data.objects.remove(s, do_unlink=True)
    
    for m in bpy.data.meshes:
        if m.users == 0:
            bpy.data.meshes.remove(m, do_unlink=True)               

def main():
    args, remainder = btb.parse_blendtorch_args()
    np.random.seed(args.btseed)
    cam = bpy.context.scene.camera

    sshapes = None

    def pre_anim():
        nonlocal sshapes
        #bpy.ops.ptcache.free_bake_all()
        sshapes = generate_supershapes(10)

    def post_frame(anim, off, pub, cam): 
        pub.publish(
            image=off.render(), 
            frameid=anim.frameid
        )

    def post_anim(anim):
        nonlocal sshapes
        remove_supershapes(sshapes)        
        sshapes = None

    pub = btb.DataPublisher(args.btsockets['DATA'], args.btid)

    bpy.context.scene.rigidbody_world.time_scale = 100

    cam = btb.Camera()
    off = btb.OffScreenRenderer(camera=cam, mode='rgb')
    off.set_render_style(shading='RENDERED', overlays=False)

    anim = btb.AnimationController()
    anim.pre_animation.add(pre_anim)
    anim.post_frame.add(post_frame, anim, off, pub, cam)
    anim.post_animation.add(post_anim, anim)
    anim.play(frame_range=(1,3), num_episodes=-1, use_animation=True)

main()