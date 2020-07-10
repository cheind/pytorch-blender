import zmq
import pickle
import io
import numpy as np
import weakref
from torch.utils import data

class InputChannelBase:
    def __init__(self, is_stream):
        self.is_stream = is_stream

class BlenderInputChannel(InputChannelBase):
    '''Base class for reading from blender publishers.'''

    def __init__(self, addresses, recorder=None, queue_size=10):
        super().__init__(is_stream=True)
        self.addresses = addresses
        self.recorder = recorder
        self.queue_size = queue_size
        self.is_stream = True
        self.ctx = None

    def recv(self, index=0, timeoutms=None):
        '''Receive from Blender instances.
        
        Receives and unpickles the next message.

        Kwargs
        ------
        index: int
            Index to read, ignored.
        timeoutms: int
            Timeout in milliseconds for the next data bundle to arrive.
            -1 to wait forever.
        '''
        if self.ctx is None:
            # Lazly creating file object here to ensure PyTorch 
            # multiprocessing compatibility (i.e num_workers > 0)
            self._create()

        socks = dict(self.poller.poll(timeoutms))
        assert self.s in socks, 'No response within timeout interval.'

        if self.recorder:
            data = self.s.recv()
            self.recorder.save(data, is_pickled=True)
            return pickle.loads(data)
        else:
            return self.s.recv_pyobj()

    def _create(self):
        '''Create communication channels.
        Note, PyTorch dataloaders do not offer a principled way to
        cleanly shut down datasets (and thus blendtorch receivers).
        Therefore we register a finalizer through weakref to cleanup.
        '''
        self.ctx = zmq.Context()
        self.s = self.ctx.socket(zmq.PULL)
        self.s.setsockopt(zmq.RCVHWM, self.queue_size)
        self.poller = zmq.Poller()
        self.poller.register(self.s, zmq.POLLIN)
        for addr in self.addresses:
            self.s.connect(addr)
        weakref.finalize(self, self._destroy)

    def _destroy(self):
        if self.s is not None:
            self.s.close()
            self.s = None
            self.ctx = None    

class FileInputChannel(InputChannelBase):
    '''Base class to read from previously recorded messages.'''

    def __init__(self, record_path):
        super().__init__(is_stream=False)
        self.record_path = record_path
        self.file = None
        self._read_header()
        
    def __len__(self):
        return self.num_messages

    def recv(self, index=0, timeoutms=-1):
        ''''Read message associated with index from disk. 
    
        Kwargs
        ------
        index: int
            Index to read.
        timeoutms: int
            Timeout in milliseconds, ignored.
        '''
        if self.file is None:
            # Lazly creating file object here to ensure PyTorch 
            # multiprocessing compatibility (i.e num_workers > 0)
            self._create()
            
        self.file.seek(self.offsets[index])
        return self.unpickler.load()

    def _create(self):
        '''Create communication channels.
        Note, PyTorch dataloaders do not offer a principled way to
        cleanly shut down datasets (and thus blendtorch receivers).
        Therefore we register a finalizer through weakref to cleanup.
        '''  
        self.file = io.open(self.record_path, 'rb')
        self.unpickler = pickle.Unpickler(self.file)
        weakref.finalize(self, self._destroy)
    
    def _read_header(self):
        with io.open(self.record_path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            self.offsets = unpickler.load()
            m = np.where(self.offsets==-1)[0]
            if len(m)==0:
                self.num_messages = len(self.offsets)
            else:
                self.num_messages = m[0]


    def _destroy(self):
        if self.file is not None:
            self.file.close()
            self.file = None

class BlenderOutputChannel:
    def __init__(self, addresses):
        self.addresses = addresses
        self.ctx = zmq.Context()

        def make_socket(addr):
            s = self.ctx.socket(zmq.PAIR)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(addr)
            return s

        self.socks = [make_socket(addr) for addr in addresses]

    def publish(self, btids=None, **kwargs):
        '''Publish a message to one or more Blender instances.

        Params
        ------
        btids : integer or iterable-like, optional
            Selects one or more Blender instances to send message to.
            If None sends message to all instances.
        kwargs: dict
            Message to send.
        '''
        if btids is None:
            btids = list(range(len(self.addresses)))
        elif isinstance(btids, int):
            btids = [btids]
        else:
            btids = list(btids)
        
        data = pickle.dumps(kwargs)
        for idx in btids:
            self.socks[idx].send(data)
        
