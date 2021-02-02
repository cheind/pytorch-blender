import bpy
import bmesh
import numpy as np
from mathutils import Vector

def find_first_view3d():
    '''Helper function to find first space view 3d and associated window region.

    The three returned objects are useful for setting up offscreen rendering in
    Blender.
    
    Returns
    -------
    area: object
        Area associated with space view.
    window: object
        Window region associated with space view.
    space: bpy.types.SpaceView3D
        Space view.
    
    '''
    areas = [a for a in bpy.context.screen.areas if a.type == 'VIEW_3D']
    assert len(areas) > 0
    area = areas[0]
    region = sorted([r for r in area.regions if r.type == 'WINDOW'], key=lambda x:x.width, reverse=True)[0]        
    spaces = [s for s in areas[0].spaces if s.type == 'VIEW_3D']
    assert len(spaces) > 0
    return area, spaces[0], region

def object_coordinates(*objs, depsgraph=None):
    '''Returns XYZ object coordinates of all objects in positional *args.
    
    Params
    ------
    objs: list-like of bpy.types.Object
        Object to return vertices for
    depsgraph: bpy.types.Depsgraph, None
        Dependency graph

    Returns
    -------
    xyz: Nx3 array
        World coordinates of object vertices
    '''

    # To be on the safe side, we use the evaluated object after 
    # all modifiers etc. applied (done internally by bmesh)
    
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    xyz = []
    for obj in objs:
        eval_obj = obj.evaluated_get(dg)
        xyz_obj = [v.co for v in eval_obj.data.vertices]
        xyz.extend(xyz_obj)
    return np.stack(xyz)

def world_coordinates(*objs, depsgraph=None):
    '''Returns XYZ world coordinates of all objects in positional *args.
    
    Params
    ------
    objs: list-like of bpy.types.Object
        Object to return vertices for
    depsgraph: bpy.types.Depsgraph, None
        Dependency graph

    Returns
    -------
    xyz: Nx3 array
        World coordinates of object vertices
    '''

    # To be on the safe side, we use the evaluated object after 
    # all modifiers etc. applied (done internally by bmesh)
    
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    xyz = []
    for obj in objs:
        eval_obj = obj.evaluated_get(dg)
        xyz_obj = [(eval_obj.matrix_world @ v.co) for v in eval_obj.data.vertices]
        xyz.extend(xyz_obj)
    return np.stack(xyz)

def bbox_world_coordinates(*objs, depsgraph=None):
    '''Returns XYZ world coordinates of all bounding box corners of each object in *objs.
    
    Params
    ------
    objs: list-like of bpy.types.Object
        Object to return vertices for
    depsgraph: bpy.types.Depsgraph, None
        Dependency graph

    Returns
    -------
    xyz: Nx3 array
        World coordinates of object vertices
    '''

    # To be on the safe side, we use the evaluated object after 
    # all modifiers etc. applied (done internally by bmesh)
    
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    xyz = []
    for obj in objs:
        eval_obj = obj.evaluated_get(dg)
        xyz_obj = [(eval_obj.matrix_world @ Vector(c)) for c in eval_obj.bound_box]
        xyz.extend(xyz_obj)
    return np.stack(xyz)


def hom(x, v=1.):
    '''Convert to homogeneous coordinates in the last dimension.'''
    return np.concatenate(
        (x, np.full((x.shape[0],1), v, dtype=x.dtype)),
        -1
    )

def dehom(x):
    '''Return de-homogeneous coordinates by perspective division.'''
    return x[...,:-1] / x[...,-1:]

def random_spherical_loc(radius_range=None, theta_range=None, phi_range=None):
    '''Return random locations on sphere.
    
    Params
    ------
    radius_range: tuple
        min/max radius of sphere. Defaults to (1,1)
    theta: tuple
        min/max longitudinal range. Defaults to (0, pi)
    phi: tuple
        min/max latitudinal range. Defaults to (0, 2*pi)
    
    Returns
    -------
    xyz : array
        location on sphere
    '''
    if radius_range is None:
        radius_range = (1,1)
    if theta_range is None:
        theta_range = (0, np.pi)
    if phi_range is None:
        phi_range = (0, 2*np.pi)

    # Not really uniform on sphere, but fine for us.
    r = np.random.uniform(radius_range[0], radius_range[1]) # radii
    t = np.random.uniform(theta_range[0], theta_range[1]) # inclination
    p = np.random.uniform(phi_range[0], phi_range[1]) # azimuth

    return np.array([
        np.sin(t)*np.cos(p),
        np.sin(t)*np.sin(p),
        np.cos(t)
    ])*r

def compute_object_visibility(obj, cam, N=25, scene=None, view_layer=None, dist=None):
    '''Computes object visibility using Monte Carlo ray-tracing.'''
    scene = scene or bpy.context.scene
    vl = view_layer or bpy.context.view_layer    
    src = cam.bpy_camera.matrix_world.translation
    dist = dist or 1.70141e+38
    
    caminv = cam.bpy_camera.matrix_world.inverted()

    ids = np.random.choice(len(obj.data.vertices), size=N)
    vis = 0
    for idx in ids:
        dst_world = obj.matrix_world @ obj.data.vertices[idx].co
        d = (dst_world-src).normalized()
        dst_cam = caminv @ dst_world
        if dst_cam.z <= 0. and np.isfinite(d).all(): # view towards neg. z            
            res,x,n,face,object,m = scene.ray_cast(vl, src, d, distance=dist)
            if res and object==obj:
                vis += 1
            del object,m,x,n,res
        del d, dst_world, dst_cam
    return vis / N

def scene_stats():
    '''Returns debug information on the current scene.'''
    stats = {}
    for attr in dir(bpy.data):
        if isinstance(attr, bpy.types.Collection):
            objs = getattr(bpy.data, attr).all_objects
            if len(objs) == 0:
                continue
            orphaned = [o for o in objs if o.users == 0]
            active = [o for o in objs if o.users > 0]
            stats[attr] = (len(active), len(orphaned))
    return stats

