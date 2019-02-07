import torch
import torch.utils.data as data
import zmq
from subprocess import Popen
import numpy as np
import sys
import atexit

class BlenderLaunch():
    def __init__(self, num_instances=3, start_port=11000, bind_addr='127.0.0.1', scene='scene.blend', instance_args=None, prot='tcp'):
        self.num_instances = num_instances
        self.start_port = start_port
        self.bind_addr = bind_addr
        self.prot = prot
        self.scene = scene
        self.instance_args = instance_args
        if instance_args is None:
            self.instance_args = [[] for _ in range(num_instances)]
        assert num_instances > 0
        assert len(self.instance_args) == num_instances

    def __enter__(self):
        ports = list(range(self.start_port, self.start_port + self.num_instances))
        self.addresses = [f'{self.prot}://{self.bind_addr}:{p}' for p in ports]
        add_args = [' '.join(a) for a in self.instance_args]

        self.processes = [Popen(f'blender {self.scene} --background --python blender.py -- {addr} {args}', 
            shell=False,
            stdin=None, 
            stdout=open(f'./tmp/out_{idx}.txt', 'w'), 
            stderr=open(f'./tmp/err_{idx}.txt', 'w'), 
            close_fds=True) for idx, (addr, args) in enumerate(zip(self.addresses, add_args))]
        
        ctx = zmq.Context()
        self.s = ctx.socket(zmq.SUB)
        self.s.setsockopt(zmq.SUBSCRIBE, b'')
        self.s.setsockopt(zmq.RCVHWM, 20)
        [self.s.connect(addr) for addr in self.addresses]

        self.poller = zmq.Poller()
        self.poller.register(self.s, zmq.POLLIN)

        return self

    def recv(self, timeoutms=-1):
        socks = dict(self.poller.poll(timeoutms))
        assert self.s in socks, 'No response within timeout interval.'
        return self.s.recv_pyobj()

    def __exit__(self, exc_type, exc_value, exc_traceback):        
        [p.terminate() for p in self.processes]
        assert not any([p.poll() for p in self.processes]), 'Blender instance still open'
        print('Blender instances closed')

class BlenderDataset:
    def __init__(self, blender_launcher, transforms=None, timeoutms=10000, dslen=sys.maxsize):
        self.blender_launcher = blender_launcher
        self.transforms = transforms
        self.timeoutms = timeoutms
        self.dslen = dslen

    def __len__(self):
        return self.dslen

    def __getitem__(self, idx):
        d = self.blender_launcher.recv(timeoutms=self.timeoutms)        

        x, coords = d['image'], d['xy']
        
        h,w = x.shape[0], x.shape[1]
        coords[...,1] = 1. - coords[...,1] # Blender uses origin bottom-left.        
        coords *= np.array([w,h])[None, :]

        if self.transforms:
            x = self.transforms(x)
        return x, coords, d['id']

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    with BlenderLaunch(num_instances=4, instance_args=[['-id', '0'], ['-id', '1'], ['-id', '2'], ['-id', '3']]) as bl:
        ds = BlenderDataset(bl)
        # Note, in the following num_workers must be 0
        dl = data.DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)

        for idx in range(10):
            x, coords, ids = next(iter(dl))
            print(f'Received from {ids}')

            # Drawing is the slow part ...
            fig, axs = plt.subplots(2,2,frameon=False, figsize=(16*2,9*2))
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
            axs = axs.reshape(-1)
            for i in range(x.shape[0]):
                axs[i].imshow(x[i], aspect='auto', origin='upper')
                axs[i].scatter(coords[i][:, 0], coords[i][:, 1], s=100)
                axs[i].set_axis_off()
            fig.savefig(f'tmp/output_{idx}.png')
            plt.close(fig)
