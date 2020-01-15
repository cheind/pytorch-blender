import zmq
from subprocess import Popen
import sys

class BlenderLauncher():
    '''Opens and closes Blender instances.
    
    This class is meant to be used withing a `with` block to ensure clean shutdown of background
    processes.
    '''

    def __init__(self, num_instances=3, start_port=11000, bind_addr='127.0.0.1', scene='scene.blend', script='blender.py', instance_args=None, prot='tcp'):
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
        '''
        self.num_instances = num_instances
        self.start_port = start_port
        self.bind_addr = bind_addr
        self.prot = prot
        self.scene = scene
        self.instance_args = instance_args
        self.script = script
        if instance_args is None:
            self.instance_args = [[] for _ in range(num_instances)]
        assert num_instances > 0
        assert len(self.instance_args) == num_instances

    def __enter__(self):
        ports = list(range(self.start_port, self.start_port + self.num_instances))
        self.addresses = [f'{self.prot}://{self.bind_addr}:{p}' for p in ports]
        add_args = [' '.join(a) for a in self.instance_args]

        self.processes = [Popen(f'blender {self.scene} --background --python {self.script} -- {addr} {args}', 
            shell=False,
            stdin=None, 
            stdout=open(f'/tmp/out_{idx}.txt', 'w'),
            stderr=open(f'/tmp/err_{idx}.txt', 'w'),
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
        '''Receive from Blender instances.
        
        Receives and unpickles the next message. When connected
        to multiple publishers, data is interleaved. When a maximum
        number of message (currently 20) are pending, new messages
        are dropped.

        Kwargs
        ------
        timeoutms: int
            Timeout in milliseconds for the next data bundle to arrive.
        '''
        socks = dict(self.poller.poll(timeoutms))
        assert self.s in socks, 'No response within timeout interval.'
        return self.s.recv_pyobj()

    def __exit__(self, exc_type, exc_value, exc_traceback):        
        [p.terminate() for p in self.processes]
        assert not any([p.poll() for p in self.processes]), 'Blender instance still open'
        print('Blender instances closed')
