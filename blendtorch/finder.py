import os
import subprocess
import re
import shutil
import logging
from pathlib import Path

logger = logging.getLogger('blendtorch')

def discover_blender(additional_blender_paths=None):
    '''Return Blender version as tuple (major, minor).'''

    my_env = os.environ.copy()
    if additional_blender_paths is not None:
        my_env['PATH'] = additional_blender_paths + os.pathsep + my_env['PATH']

    exe = shutil.which('blender', path=my_env['PATH'])
    if exe is None:
        logger.warning('Could not find Blender.')
        return None
    exe = Path(exe).resolve()
    
    p = subprocess.Popen(f'"{exe}" --version', 
            shell=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env=my_env)

    out, err = p.communicate()
    errcode = p.returncode

    r = re.compile(r'Blender\s(\d+)\.(\d+)', re.IGNORECASE)
    g = re.search(r, str(out))

    version = (None, None)
    if errcode == 0 and g is not None:
        version = (int(g[1]),int(g[2]))
    else:
        logger.warning('Failed to parse Blender version.')
        return None

    return {'path':exe, 'major':version[0], 'minor':version[1]}      

if __name__ == '__main__':
    print(discover_blender('c:/dev/blender279'))