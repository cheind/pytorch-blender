import bpy
import bmesh
import numpy as np

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


def hom(x, v=1.):
    '''Convert to homogeneous coordinates in the last dimension.'''
    return np.concatenate(
        (x, np.full((x.shape[0],1), v, dtype=x.dtype)),
        -1
    )

def dehom(x):
    '''Return de-homogeneous coordinates by perspective division.'''
    return x[...,:-1] / x[...,-2:-1]