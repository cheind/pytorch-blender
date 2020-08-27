import zmq
import time
import os
import sys

from .constants import DEFAULT_TIMEOUTMS

class DuplexChannel:
    '''Provides generic bidirectional communication with a single PyTorch instance.'''
    def __init__(self, address, btid=None, lingerms=0, timeoutms=DEFAULT_TIMEOUTMS):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PAIR)
        self.sock.setsockopt(zmq.LINGER, lingerms)
        self.sock.setsockopt(zmq.RCVHWM, 10)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        self.sock.setsockopt(zmq.SNDTIMEO, timeoutms)
        self.sock.setsockopt(zmq.RCVTIMEO, timeoutms)
        self.sock.bind(address)

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)
        self.btid = btid

    def recv(self, timeoutms=None):
        '''Return next message or None.
        
        Kwargs
        ------
        timeoutms: int
            Max time to spend waiting for messages. If None, blocks until
            at least one message is available.

        Returns
        -------
        msg: dict, None
            Message received or None.        
        '''
        socks = dict(self.poller.poll(timeoutms))
        if self.sock in socks:
            return self.sock.recv_pyobj()
        else:
            return None

    def send(self, **kwargs):
        '''Send a message to remote Blender process.

        Automatically attaches the process identifier `btid` and
        a unique message id `btmid` to the dictionary.

        Params
        ------
        kwargs: dict
            Message to send.

        Returns
        -------
        messageid: integer
            Message id attached to dictionary
        '''
        mid = int.from_bytes(os.urandom(4), sys.byteorder)
        data = {
            'btid':self.btid, 
            'btmid': mid, 
            **kwargs
        }
        self.sock.send_pyobj(data)
        return mid