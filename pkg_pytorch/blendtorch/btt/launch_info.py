import json
from contextlib import ExitStack

class LaunchInfo:
    '''Holds information about running Blender instances.
    
    Attributes
    ----------
    addresses: dict
        Dictionary of spawned addresses grouped by socket name.
    commands: list
        List of command line arguments used to spawn Blender instances.
    processes: list
        List of running spawned processes. Only populated when
        locally launched through BlenderLauncher, otherwise None.
    '''
    def __init__(self, addresses, commands, processes=None):
        self.addresses = addresses
        self.processes = processes
        self.commands = commands

    @staticmethod
    def save_json(file, launch_info):
        '''Save launch information in JSON format.

        Useful if you want to reconnect to running instances from a different location.
        This will only serialize addresses and commands.

        Params
        ------
        file: file-like object, string, or pathlib.Path
            The file to save to.
        launch_info: LaunchInfo
            The launch information to save.
        '''
        with ExitStack() as stack:
            if hasattr(file, 'write'):
                fp = stack.enter_context(nullcontext(file))
            else:
                fp = stack.enter_context(open(file, 'w'))
            json.dump({'addresses': launch_info.addresses, 'commands': launch_info.commands}, fp, indent=4)

    @staticmethod
    def load_json(file):
        '''Load launch information from JSON format.

        Params
        ------
        file: file-like object, string, or pathlib.Path
            The file to read from.

        Returns
        -------
        launch_info: LaunchInfo
            Restored launch information
        '''
        with ExitStack() as stack:
            if hasattr(file, 'read'):
                fp = stack.enter_context(nullcontext(file))
            else:
                fp = stack.enter_context(open(file, 'r'))
            data = json.load(fp)
        return LaunchInfo(data['addresses'], data['commands'])