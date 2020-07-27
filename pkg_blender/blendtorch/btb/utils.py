import bpy

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