import zmq

class Subscriber:
    '''Base class for reading from blender publishers.'''

    def __init__(self):
        self.ctx = zmq.Context()
        self.s = self.ctx.socket(zmq.SUB)
        self.s.setsockopt(zmq.SUBSCRIBE, b'')
        self.s.setsockopt(zmq.RCVHWM, 20)
        self.poller = zmq.Poller()
        self.poller.register(self.s, zmq.POLLIN)

    def connect(self, addresses):
        if not isinstance(addresses, list):
            addr = [addr]
        for addr in addresses:
            self.s.connect(addr)

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

    def __call__(self, timeoutms=-1):
        return self.recv(timeoutms)
