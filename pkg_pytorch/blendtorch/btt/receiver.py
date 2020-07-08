import zmq
import pickle
import io
import numpy as np
import sys

class BlenderReceiver:
    '''Base class for reading from blender publishers.'''

    def __init__(self, recorder=None, batch_size=10, num_messages=sys.maxsize):
        self.recorder = recorder
        self.batch_size = batch_size        
        self.num_messages = num_messages

    def connect(self, addresses):
        if not isinstance(addresses, list):
            addresses = [addresses]
        for addr in addresses:
            self.s.connect(addr)

    def __len__(self):
        return self.num_messages

    def recv(self, index=0, timeoutms=-1):
        '''Receive from Blender instances.
        
        Receives and unpickles the next message.

        Kwargs
        ------
        index: int
            Index to read, ignored.
        timeoutms: int
            Timeout in milliseconds for the next data bundle to arrive.
        '''
        socks = dict(self.poller.poll(timeoutms))
        assert self.s in socks, 'No response within timeout interval.'

        if self.recorder:
            data = self.s.recv()
            self.recorder.save(data, is_pickled=True)
            return pickle.loads(data)
        else:
            return self.s.recv_pyobj()

    def __enter__(self):
        self.ctx = zmq.Context()
        self.s = self.ctx.socket(zmq.PULL)
        self.s.setsockopt(zmq.RCVHWM, self.batch_size)
        self.poller = zmq.Poller()
        self.poller.register(self.s, zmq.POLLIN)
        return self

    def __exit__(self, *args):
        self.s.close()


class FileReceiver:
    '''Base class to read from previously recorded messages.'''

    def __init__(self, record_path):
        self.record_path = record_path
        self.num_messages = 0
        
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
        self.file.seek(self.offsets[index])
        return self.unpickler.load()

    def __enter__(self):
        self.file = io.open(self.record_path, 'rb')
        self.unpickler = pickle.Unpickler(self.file)
        self.offsets = self.unpickler.load()
        m = np.where(self.offsets==-1)[0]
        if len(m)==0:
            self.num_messages = len(self.offsets)
        else:
            self.num_messages = m[0] 
        return self

    def __exit__(self, *args):
        self.file.close()
