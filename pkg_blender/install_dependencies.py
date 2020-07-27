'''Install Blender dependencies.

Meant to be run ONCE via blender as follows
`blender --background --python pkg_blender/install_dependencies.py`
'''

import bpy
import subprocess

def install(name, upgrade=True):
    try:
        cmd = [bpy.app.binary_path_python, '-m', 'pip', 'install', name, '--user']
        if upgrade:
            cmd.append('--upgrade')
        output = subprocess.check_output(cmd)
        print(output)   
    except subprocess.CalledProcessError as e:
        print(e.output)

def main():
    install('pip')
    install('PyOpenGL')
    install('Pillow')
    install('PyZMQ')
    bpy.ops.wm.quit_blender()

main()