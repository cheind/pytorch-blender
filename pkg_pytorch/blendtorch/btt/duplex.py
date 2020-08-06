import zmq
import pickle
import time

from .constants import DEFAULT_TIMEOUTMS

class DuplexChannel:
    '''Provides generic bidirectional communication with a single Blender instance.
    '''

    def __init__(self, address, lingerms=0):
        self.address = address
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PAIR)
        self.sock.setsockopt(zmq.LINGER, lingerms)
        self.sock.connect(address)

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

    def send(self, data):
        '''Send a message to remote Blender process.

        Params
        ------
        data: object
            Message to send.
        '''        
        self.sock.send_pyobj(data)

    def recv(self, timeoutms=None):
        '''Receive from PyTorch instance.
                
        Receives and unpickles all available messages, returned as list.
        
        Kwargs
        ------
        timeoutms: int
            Max time to spend waiting for messages. If None, blocks until
            at least one message is available.

        Returns
        -------
        msgs: list(object), None
            List of received messages within `timeoutms`.
        '''
        t = time.time()
        leftms = timeoutms        
        messages = []
        while leftms is None or leftms >= 0:
            socks = dict(self.poller.poll(leftms))
            if self.sock in socks:
                messages.append(self.sock.recv_pyobj())
            if leftms is None:
                break
            now = time.time()
            leftms -= (now - t)*1e3
            t = now
        return messages