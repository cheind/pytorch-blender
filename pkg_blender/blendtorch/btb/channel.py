import zmq
import bpy
import time

class BlenderOutputChannel:
    '''Data publisher to be used within Blender scripts.'''

    def __init__(self, out_address, btid):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(out_address)
        self.btid = btid
        
    def publish(self, **kwargs):
        '''Publish message.

        This method will implicitly add bendtorch instance id 
        `btid` to the send dictionary.

        Params
        ------
        kwargs: optional
            Dictionary of pickable objects composing the message.
        '''

        data = dict(btid=self.btid)
        data.update(kwargs)
        self.sock.send_pyobj(data)

class BlenderInputChannel:
    def __init__(self, in_address):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PAIR)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(in_address)
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

    def recv(self, timeoutms=None):
        '''Receive from PyTorch instance.
        
        Receives and unpickles all available messages, returned as list.

        Kwargs
        ------
        timeoutms: int
            Max time to spend waiting for messages. If None, blocks until
            at least one message is available.
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
            leftms -= (now - t)
            t = now

        if len(messages) == 0:
            return None
        else:
            return messages