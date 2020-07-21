import zmq

class RemoteEnv:
    def __init__(self, address):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(address)

    def reset(self):
        return self._reqrep('reset', action=None)

    def step(self, action):
        return self._reqrep('step', action=action)
        pass

    def _reqrep(self, cmd, action=None):
        self.socket.send_pyobj((cmd, action))
        return self.socket.recv_pyobj()