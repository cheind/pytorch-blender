from .animation import AnimationController
import zmq


class BaseEnv:
    '''Environment base class, based on the model of OpenAI Gym.'''

    def __init__(self, address):
        self.animation = AnimationController()
        self.animation.pre_frame.add(self.pre_frame)
        self.animation.post_frame.add(self.post_frame)
        self.prev_action = None
        self.obs = None
        self.reward = None
        
        # self.context = zmq.Context()
        # self.socket = context.socket(zmq.REP)
        # self.socket.connect(address)
        # self.poller = zmq.Poller()
        # self.poller.register(self.socket, zmq.POLLIN)
        # self.state = 'IDLE'

    def pre_frame(self):
        if self.animation.frameid > self.frame_range[0]:
            self.prev_action = self.controlfn(self.prev_action, self.)

        self.
        pass

    def post_frame(self):
        if self.animation.frameid == self.frame_range[0]:

        pass

    def reset(self, controlfn, frame_range):        
        self.animation.cancel()
        self.controlfn = controlfn
        self.frame_range = frame_range
        self.prev_action = None
        self.done = False
        self.animation.play(frame_range, num_episodes=1)
        

