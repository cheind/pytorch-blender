import bpy
import sys

def add_scene_dir_to_path():
    '''Adds directory of scene file to Python path'''
    p = bpy.path.abspath("//")
    if p not in sys.path:
        sys.path.append(p)
