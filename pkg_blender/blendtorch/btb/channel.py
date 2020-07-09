import zmq
import bpy

class BlenderOutputChannel:
    '''Data publisher to be used within Blender scripts.'''

    def __init__(self, bind_address, btid):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        self.sock.bind(bind_address)
        self.btid = btid
        
    def publish(self, **kwargs):
        data = dict(btid=self.btid, frameid=bpy.context.scene.frame_current)
        data.update(kwargs)
        self.sock.send_pyobj(data)