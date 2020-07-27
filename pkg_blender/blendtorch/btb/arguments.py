import sys
import argparse
import os

def parse_blendtorch_args(argv=None):
    '''Parses blendtorch instance parameters and returns the remainder arguments.
    
    This method is intended to be used with Blender instances launched via 
    `btt.BlenderLauncher`. It parses specific command line arguments that
     - identify the Blender process `btid`,
     - provide a random number seed `btseed`, and
     - lists a number of named socket addresses to connect to.

    This script parses command-line arguments after a special end of
    command line element `--`.
    
    Params
    ------
    argv: list-like, None
        The command line arguments to be parsed.

    Returns
    -------
    args: argparse.Namespace
        The parsed command line arguments
    remainder: list-like
        The remaining command line arguments.
    '''    
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