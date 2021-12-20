import os
import subprocess
import re
import shutil
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger('blendtorch')

script = r'''
import zmq

'''

def discover_blender(additional_blender_paths=None):
    '''Return Blender info as dict with keys `path`, `major`, `minor`.'''

    my_env = os.environ.copy()
    if additional_blender_paths is not None:
        my_env['PATH'] = additional_blender_paths + os.pathsep + my_env['PATH']

    # Determine path

    bpath = shutil.which('blender', path=my_env['PATH'])
    if bpath is None:
        logger.warning('Could not find Blender.')
        return None
    else:
        logger.debug(f'Discovered Blender in {bpath}')
    # Using absolute instead of resolve to not follow symlinks (snap issue on linux)
    bpath = Path(bpath).absolute()
    
    p = subprocess.Popen(f'"{bpath}" --version', 
            shell=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env=my_env)

    out, err = p.communicate()
    errcode = p.returncode

    # Determine version

    r = re.compile(r'Blender\s(\d+)\.(\d+)', re.IGNORECASE)
    g = re.search(r, str(out))

    version = (None, None)
    if errcode == 0 and g is not None:
        version = (int(g[1]),int(g[2]))
    else:
        logger.warning('Failed to parse Blender version.')
        return None

    # Check if a minimal Python script works 
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as fp:
        fp.write(script)
    
    p = subprocess.Popen(f'"{bpath}" --background --python-use-system-env --python-exit-code 255 --python {fp.name}', 
        shell=True,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        env=my_env)
    out, err = p.communicate()
    errcode = p.returncode
    os.remove(fp.name)

    if errcode != 0:
        logger.warning('Failed to run minimal Blender script; ensure Python requirements are installed.')
        return None

    return {'path':bpath, 'major':version[0], 'minor':version[1]}      

def _main():
    print(discover_blender())

if __name__ == '__main__':
    _main()