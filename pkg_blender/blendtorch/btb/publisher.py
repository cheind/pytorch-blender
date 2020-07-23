import bpy
import zmq

class DataPublisher:
    '''Data publisher'''

    def __init__(self, bind_address, btid):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(bind_address)
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

        data = {'btid':self.btid, **kwargs}
        self.sock.send_pyobj(data)