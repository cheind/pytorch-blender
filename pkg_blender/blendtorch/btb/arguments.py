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

    addrsplit = lambda x: tuple(x.split('='))

    parser = argparse.ArgumentParser()    
    parser.add_argument('-btid', type=int, help='Identifier for this Blender instance')    
    parser.add_argument('-btseed', type=int, help='Random number seed')
    parser.add_argument("-btsockets",
        metavar="NAME=ADDRESS",
        nargs='*',
        type=addrsplit,
        help="Set a number of named address pairs.")        
    args, remainder = parser.parse_known_args(argv)
    args.btsockets = dict(args.btsockets)
    return args, remainder