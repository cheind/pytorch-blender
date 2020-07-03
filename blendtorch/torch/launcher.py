from subprocess import Popen
import sys
import os
import logging

from .finder import discover_blender

logger = logging.getLogger('blendtorch')

class BlenderLauncher():
    '''Opens and closes Blender instances.
    
    This class is meant to be used withing a `with` block to ensure clean shutdown of background
    processes.
    '''

    def __init__(self, num_instances=3, start_port=11000, bind_addr='127.0.0.1', scene='scene.blend', script='blender.py', instance_args=None, prot='tcp', blend_path=None):
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
        if instance_args is None:
            self.instance_args = [[] for _ in range(num_instances)]
        assert num_instances > 0
        assert len(self.instance_args) == num_instances

        self.blender_info = discover_blender(self.blend_path)

    def __enter__(self):
        ports = list(range(self.start_port, self.start_port + self.num_instances))
        self.addresses = [f'{self.prot}://{self.bind_addr}:{p}' for p in ports]
        
        # Add blendtorch instance identifiers to instances        
        [iargs.append(f'-btid {idx}') for idx,iargs in enumerate(self.instance_args)]      
        [iargs.append(f'-bind-address {addr}') for addr,iargs in zip(self.addresses,self.instance_args)]
        args = [' '.join(a) for a in self.instance_args]
        
        if self.blender_info is None:
            logger.warning('Launching Blender failed;')
            raise ValueError('Blender not found or misconfigured.') 
        else:
            logger.info(f'Blender found {self.blender_info["path"]} version {self.blender_info["major"]}.{self.blender_info["minor"]}')

        # Update python path to find this package within Blender python.
        env = os.environ.copy()  
        this_package_path = os.pathsep + os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] += this_package_path
        else:
            env['PYTHONPATH'] = this_package_path

        self.processes = [Popen(f'"{self.blender_info["path"]}" {self.scene} --python-use-system-env --log-file ./tmp/log_{idx}.txt --python {self.script} -- {args}', 
            shell=False,
            stdin=None, 
            stdout=open(f'./tmp/out_{idx}.txt', 'w'),
            stderr=None, 
            close_fds=True,
            env=env) for idx,args in enumerate(args)]

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):    
        [p.terminate() for p in self.processes]        
        assert not any([p.poll() for p in self.processes]), 'Blender instance still open'
        logger.info('Blender instances closed')
        