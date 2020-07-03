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

# need to start blender like this: 
# --python-use-system-env