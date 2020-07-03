import bpy
import subprocess

def install(name):
    try:
        output = subprocess.check_output([bpy.app.binary_path_python, '-m', 'pip', 'install', name, '--user'])     
        print(output)   
    except subprocess.CalledProcessError as e:
        print(e.output)

install('PyOpenGL')
install('Pillow')
install('PyZMQ')

# need to start blender like this: 
# "c:\Program Files\Blender Foundation\Blender 2.83\blender.exe" --python-use-system-env --python blender28\simple.py