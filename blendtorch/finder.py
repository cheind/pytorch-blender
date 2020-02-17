import os
import subprocess
import re

def blender_version(blender_path=None):
    '''Return Blender version as tuple (major, minor).'''

    my_env = os.environ.copy()
    if blender_path is not None:
        my_env['PATH'] = blender_path + os.pathsep + my_env['PATH']
    
    p = subprocess.Popen(f'blender --version', 
            shell=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env=my_env)

    out, err = p.communicate()
    errcode = p.returncode

    r = re.compile(r'Blender\s(\d+)\.(\d+)', re.IGNORECASE)
    g = re.search(r, str(out))

    if errcode == 0 and g is not None:
        return (int(g[1]),int(g[2]))
    else:
        return None

if __name__ == '__main__':
    print(blender_version('c:/dev/blender279'))