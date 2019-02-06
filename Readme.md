# pytorch-blender

Seamless integration of Blender renderings into pytorch for deep learning from artificial visual data. This repository contains a demonstration of a pytorch dataset harvesting ever changing data from a Blender instances.



>>> bpy.ops.render.render()
{'FINISHED'}

>>> pixels = np.array(bpy.data.images['Viewer Node'].pixels)
Traceback (most recent call last):
  File "<blender_console>", line 1, in <module>
NameError: name 'np' is not defined

>>> import numpy as np
>>> pixels = np.array(bpy.data.images['Viewer Node'].pixels)
Traceback (most recent call last):
  File "<blender_console>", line 1, in <module>
KeyError: 'bpy_prop_collection[key]: key "Viewer Node" not found'

>>> bpy.ops.render.render()
{'FINISHED'}

>>> pixels = np.array(bpy.data.images['Viewer Node'].pixels)
>>> len(pixels)
8294400

>>> width = bpy.context.scene.render.resolution_x 
>>> height = bpy.context.scene.render.resolution_y
>>> image = pixels.reshape(height,width,4)