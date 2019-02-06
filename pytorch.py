import torch
import torch.utils.data as data
import zmq
from subprocess import Popen
import numpy as np
import sys
import atexit

class BlenderDataset:
    def __init__(self, num_worker=3, transforms=None):
        self.transforms = transforms
        self.setup_communication(num_worker)

    def setup_communication(self, num_worker):
        ports = list(range(11000,11000+num_worker))
        addresses = [f'tcp://127.0.0.1:{p}' for p in ports]
        self.processes = [Popen(f'blender scene.blend --python blender.py -- {addr}', 
            shell=True,
            stdin=None, 
            stdout=None, 
            stderr=None, 
            close_fds=True) for addr in addresses]
        atexit.register(self.cleanup)
        print(addresses)

        ctx = zmq.Context()
        self.s = ctx.socket(zmq.SUB)
        self.s.setsockopt(zmq.SUBSCRIBE, b'')
        self.s.setsockopt(zmq.RCVHWM, 20)
        [self.s.connect(addr) for addr in addresses]

        self.poller = zmq.Poller()
        self.poller.register(self.s, zmq.POLLIN)

    def cleanup(self):
        [p.terminate() for p in self.processes]
    
    def __len__(self):
        return sys.maxsize

    def __getitem__(self, idx):
        socks = dict(self.poller.poll(20000))
        assert self.s in socks, 'No response within timeout interval.'
        
        x = self.s.recv_pyobj()
        if self.transforms:
            x = self.transforms(x)
        return x


ds = BlenderDataset(2)
print(ds[0].shape)

ds.cleanup()