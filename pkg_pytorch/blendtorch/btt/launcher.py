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
    
    This class is meant to be used withing a `with` block to ensure clean shutdown of background
    processes.
    '''

    def __init__(self, num_instances=3, start_port=11000, bind_addr='127.0.0.1', scene='scene.blend', script='blender.py', instance_args=None, prot='tcp', blend_path=None, seed=None):
        '''Initialize instance.
        
        Kwargs
        ------
        num_instances: int (default=3)
            How many Blender instances to create
        start_port : int (default=11000)
            Start of port range for publisher sockets
        bind_addr : string (default='127.0.0.1')
            Address to bind publisher sockets
        scene : string (default='scene.blend')
            Scene file to be processed by Blender instances
        instance_args : array (default=None)
            Additional arguments per instance to be passed as command
            line arguments.
        script: string
            Script to be called from Blender
        blend_path: string
            Additional paths to look for Blender
        '''
        self.num_instances = num_instances
        self.start_port = start_port
        self.bind_addr = bind_addr
        self.prot = prot
        self.scene = scene
        self.instance_args = instance_args
        self.script = script
        self.blend_path = blend_path
        self.launch_info = None
        self.seed = seed
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

    def __enter__(self):
        assert self.launch_info is None, 'Already launched.'
        ports = list(range(self.start_port, self.start_port + self.num_instances))
        addresses = [f'{self.prot}://{self.bind_addr}:{p}' for p in ports]
        if self.seed is None:
            seeds = np.random.randint(0, 10000, dtype=int, size=self.num_instances)
        else:
            seeds = [self.seed + i for i in range(self.num_instances)]
        
        # Add blendtorch instance identifiers to instances        
        [iargs.append(f'-btid {idx}') for idx,iargs in enumerate(self.instance_args)]      
        [iargs.append(f'-btseed {seed}') for seed,iargs in zip(seeds,self.instance_args)]
        [iargs.append(f'-bind-address {addr}') for addr,iargs in zip(addresses,self.instance_args)]
        args = [' '.join(a) for a in self.instance_args]
       
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

    def __exit__(self, exc_type, exc_value, exc_traceback):    
        [p.terminate() for p in self.launch_info.processes]
        assert not any([p.poll() for p in self.launch_info.processes]), 'Blender instance still open'
        self.launch_info = None
        logger.info('Blender instances closed')