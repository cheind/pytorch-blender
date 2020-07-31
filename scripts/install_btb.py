'''Install Blender dependencies.

Meant to be run ONCE via blender as follows
`blender --background --python scripts/install_btb.py`
'''

import bpy
import sys
import subprocess
from pathlib import Path

THISDIR = Path(__file__).parent

def install(name, upgrade=True, user=True, editable=False):
    try:
        cmd = [bpy.app.binary_path_python, '-m', 'pip', 'install']
        if upgrade:
            cmd.append('--upgrade')
        if user:
            cmd.append('--user')
        if editable:
            cmd.append('-e')
        cmd.append(name)
        output = subprocess.check_output(cmd)
        print(output)
    except subprocess.CalledProcessError as e:
        print(e.output)
        sys.exit(1)

def main():
    install('pip', upgrade=True)
    install(str(THISDIR/'..'/'pkg_blender'), editable=True)

main()