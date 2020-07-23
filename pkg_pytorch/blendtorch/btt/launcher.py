import subprocess
import sys
import os
import logging
import platform
import signal
import numpy as np

from .finder import discover_blender

logger = logging.getLogger('blendtorch')

class LaunchInfo:
    def __init__(self, addresses, processes, commands):
        self.addresses = addresses
        self.processes = processes
        self.commands = commands


class BlenderLauncher():
    '''Opens and closes Blender instances.
    
    This class is meant to be used withing a `with` block to ensure clean launch/shutdown of background processes.

    Params
    ------
    scene : str
        Scene file to be processed by Blender instances            
    script: str
        Script file to be called by Blender
    num_instances: int (default=1)
        How many Blender instances to create
    named_sockets: list-like, optional
        Descriptive names of sockets to be passed to launched instanced
        via command-line arguments '-btsockets name=tcp://address:port ...' 
        to Blender. They are also available via LaunchInfo to PyTorch.
    start_port : int (default=11000)
        Start of port range for publisher sockets
    bind_addr : str (default='127.0.0.1')
        Address to bind publisher sockets  
    proto: string (default='tcp')
        Protocol to use.      
    instance_args : array (default=None)
        Additional arguments per instance to be passed as command
        line arguments.        
    blend_path: str, optional
        Additional paths to look for Blender
    seed: integer, optional
        Optional launch seed. Each instance will be given

    Attributes
    ----------
    launch_info: LaunchInfo
        Process launch information, available after entering the
        context.
    '''

    def __init__(self, scene, script, num_instances=1, named_sockets=None, start_port=11000, bind_addr='127.0.0.1', instance_args=None, proto='tcp', blend_path=None, seed=None):
        '''Create BlenderLauncher'''
        self.num_instances = num_instances
        self.start_port = start_port
        self.bind_addr = bind_addr
        self.proto = proto
        self.scene = scene        
        self.script = script
        self.blend_path = blend_path
        self.named_sockets = named_sockets
        if named_sockets is None:
            self.named_sockets = []
        self.seed = seed
        self.instance_args = instance_args
        if instance_args is None:
            self.instance_args = [[] for _ in range(num_instances)]
        assert num_instances > 0
        assert len(self.instance_args) == num_instances

        self.blender_info = discover_blender(self.blend_path)
        if self.blender_info is None:
            logger.warning('Launching Blender failed;')
            raise ValueError('Blender not found or misconfigured.') 
        else:
            logger.info(f'Blender found {self.blender_info["path"]} version {self.blender_info["major"]}.{self.blender_info["minor"]}')

        self.launch_info = None

    def __enter__(self):
        '''Launch processes'''
        assert self.launch_info is None, 'Already launched.'
        
        addresses = {}
        addrgen = self._address_generator(self.proto, self.bind_addr, self.start_port)
        format_addr = lambda idx: ' '.join([f'{k}={v[idx]}' for k,v in addresses.items()])
        for s in self.named_sockets:
            addresses[s] = [next(addrgen) for _ in range(self.num_instances)]
        
        seed = self.seed
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max - self.num_instances)
        seeds = [seed + i for i in range(self.num_instances)]

        final_args = [[] for _ in range(self.num_instances)]
        for idx, iargs in enumerate(final_args):
            iargs.extend([
                '-btid', str(idx),
                '-btseed', str(seeds[idx]),
                '-btsockets', format_addr(idx),
            ])
            iargs.extend(self.instance_args[idx])
        args = [' '.join(a) for a in final_args]
       
        processes = []
        commands = []
        env = os.environ.copy()
        for idx,arg in enumerate(args):
            cmd = f'"{self.blender_info["path"]}" {self.scene} --python-use-system-env  --python {self.script} -- {arg}'
            p = subprocess.Popen(
                cmd,
                shell=False,
                stdin=None, 
                stdout=None,
                stderr=None,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )

            processes.append(p)
            commands.append(cmd)
            logger.info(f'Started instance: {cmd}')

        self.launch_info = LaunchInfo(addresses, processes, commands)
        return self

    def assert_alive(self):
        '''Tests if all launched process are alive.'''
        if self.launch_info is None:
            return
        codes = self._poll()
        assert all([c==None for c in codes]), f'Alive test failed. Exit codes {codes}'

    def __exit__(self, exc_type, exc_value, exc_traceback): 
        '''Terminate all processes.'''   
        [p.terminate() for p in self.launch_info.processes]
        [p.wait() for p in self.launch_info.processes]
        assert all([c!=None for c in self._poll()]), 'Not all Blender instances closed.'
        self.launch_info = None
        logger.info('Blender instances closed')

    def _address_generator(self, proto, bind_addr, start_port):
        '''Convenience to generate addresses.'''
        nextport = start_port
        while True:
            addr = f'{proto}://{bind_addr}:{nextport}'
            nextport += 1
            yield addr

    def _poll(self):
        '''Convenience to poll all processes exit codes.'''
        return [p.poll() for p in self.launch_info.processes]