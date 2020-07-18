from .animation import AnimationController
from collections import namedtuple

ControllerInfo = namedtuple('ControllerInfo', ['action', 'observation', 'reward', 'done', 'time'])

class BaseEnv:
    '''Environment base class, based on the model of OpenAI Gym.'''

    def __init__(self, controlfn):
        self.animation = AnimationController()
        self.animation.pre_frame.add(self.pre_frame)
        self.animation.post_frame.add(self.post_frame)
        self.animation.pre_animation.add(self.pre_animation)
        self.animation.post_animation.add(self.post_animation)
        self.controlfn = controlfn
        self.prev_action = None
        self._controlinfo = None

    def pre_frame(self):
        # get action using current state
        next_action = self.controlfn(self._controlinfo)
        self.apply_action(next_action)
        self.prev_action = next_action

    def post_frame(self):
        self._controlinfo = self.update_controlinfo()

    def pre_animation(self):
        self._controlinfo = None

    def post_animation(self):
        pass

    def run(self, startframe=None, endframe=None):
        self.animation.play(once=False, startframe=0, endframe=100)

