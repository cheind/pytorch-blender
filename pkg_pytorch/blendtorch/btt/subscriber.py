import zmq
import pickle

class Subscriber:
    '''Base class for reading from blender publishers.'''

    def __init__(self, recorder=None):
        self.ctx = zmq.Context()
        self.s = self.ctx.socket(zmq.PULL)
        self.s.setsockopt(zmq.RCVHWM, 10)
        self.poller = zmq.Poller()
        self.poller.register(self.s, zmq.POLLIN)
        self.recorder = recorder

    def connect(self, addresses):
        if not isinstance(addresses, list):
            addresses = [addresses]
        for addr in addresses:
            self.s.connect(addr)

    def recv(self, timeoutms=-1):
        '''Receive from Blender instances.
        
        Receives and unpickles the next message.

        Kwargs
        ------
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

    def __call__(self, timeoutms=-1):
        return self.recv(timeoutms)
