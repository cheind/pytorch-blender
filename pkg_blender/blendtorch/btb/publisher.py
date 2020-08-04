import bpy
import zmq

class DataPublisher:
    '''Publish rendered images and auxilary data.
    
    This class acts as data source for `btt.RemoteIterableDataset`. 
    Keyword arguments to `publish` must be pickle-able.

    Params
    ------
    bind_address: str
        ZMQ remote address to bind to.
    btid: integer
        blendtorch process identifier when available. Will be auto-added
        to the send dictionary. Defaults to None.
    send_hwm: integer
        Max send queue size before blocking caller of `publish`.
    '''

    def __init__(self, bind_address, btid=None, send_hwm=10, lingerms=0):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, send_hwm)
        self.sock.setsockopt(zmq.LINGER, lingerms)
        self.sock.setsockopt(zmq.IMMEDIATE, 1)
        self.sock.bind(bind_address)
        self.btid = btid
        
    def publish(self, **kwargs):
        '''Publish a message.

        This method will implicitly add bendtorch instance id  `btid` 
        to the send dictionary.

        Params
        ------
        kwargs: optional
            Dictionary of pickable objects composing the message.
        '''

        data = {'btid':self.btid, **kwargs}
        self.sock.send_pyobj(data)