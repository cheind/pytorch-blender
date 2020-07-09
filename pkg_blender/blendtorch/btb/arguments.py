import sys
import argparse
import os

def parse_blendtorch_args(argv=None):
    '''Parses blendtorch instance parameters and returns the remainder arguments.'''    
    argv = argv or sys.argv
    if '--' in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        raise ValueError('No script arguments found; missing `--`?')

    parser = argparse.ArgumentParser()    
    parser.add_argument('-btid', type=int, help='Identifier for this Blender instance')    
    parser.add_argument('-btseed', type=int, help='Random number seed')
    parser.add_argument('-bind-address', help='Address to bind to.')
    args, remainder = parser.parse_known_args(argv)
    return args, remainder