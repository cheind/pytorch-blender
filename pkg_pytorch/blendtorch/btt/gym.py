import zmq

from .constants import DEFAULT_TIMEOUTMS

class RemoteEnv:
    def __init__(self, address, timeoutms=DEFAULT_TIMEOUTMS):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(address)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.timeoutms = timeoutms

    def reset(self):
        data = self._reqrep('reset', action=None)
        return data['obs']

    def step(self, action):
        data = self._reqrep('step', action=action)
        return data['obs'], data['reward'], data['done'], data['info']

    def _reqrep(self, cmd, action=None):
        self.socket.send_pyobj((cmd, action))
        
        socks = dict(self.poller.poll(self.timeoutms))
        assert self.socket in socks, 'No response within timeout interval.'

        return self.socket.recv_pyobj()