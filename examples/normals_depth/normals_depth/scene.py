import bpy
import bmesh
import numpy as np

import supershape as sshape

SCN = bpy.context.scene


def create_sshape_mesh(shape, material=None, fake_user=True):
    new_obj = sshape.make_bpy_mesh(shape, name='sshape', coll=False, weld=True)
    new_obj.data.use_fake_user = fake_user
    new_obj.use_fake_user = fake_user
    if material is not None:
        new_obj.data.materials.append(material)
        new_obj.active_material_index = 0
    return new_obj


def prepare(n_sshapes, sshape_res=(100, 100), collection='Generated', mat='Normals', fake_user=False):
    coll = SCN.collection.children[collection]
    mat = bpy.data.materials[mat]

    sshapes = [
        create_sshape_mesh(sshape_res, material=mat, fake_user=fake_user)
        for _ in range(n_sshapes)
    ]

    for s in sshapes:
        coll.objects.link(s)
        SCN.rigidbody_world.collection.objects.link(s)
        # Rigid body settings
        s.rigid_body.enabled = True
        #s.rigid_body.collision_shape = 'CONVEX_HULL'
        s.rigid_body.collision_shape = 'BOX'
        s.rigid_body.friction = 0.7
        s.rigid_body.linear_damping = 0.3
        s.rigid_body.angular_damping = 0.4
        s.rigid_body.type = 'ACTIVE'

    return sshapes


def update_mesh(mesh, sshape_res=(100, 100)):
    params = np.random.uniform(
        low=[1.00, 1, 1, 6.0, 6.0, 6.0],
        high=[4.00, 1, 1, 10.0, 10.0, 10.0],
        size=(2, 6)
    )
    scale = np.abs(np.random.normal(0.05, 0.05, size=3))
    x, y, z = sshape.supercoords(params, shape=sshape_res)
    sshape.update_bpy_mesh(x*scale[0], y*scale[1], z*scale[2], mesh)
    mesh.location = np.random.uniform(
        low=[-0.5, -0.5, 1], high=[0.5, 0.5, 3], size=(3))
    mesh.rotation_euler = np.random.uniform(low=-np.pi, high=np.pi, size=(3))
