import bpy
from mathutils import Matrix
import numpy as np

from blendtorch import btb

class MyEnv(btb.env.BaseEnv):
    def __init__(self, agent, done_after=10):
        super().__init__(agent)
        self.cube = bpy.data.objects['Cube']
        self.count = 0
        self.done_after=done_after

    def _env_reset(self):
        self.cube.rotation_euler[2] = 0.
        self.count = 0

    def _env_prepare_step(self, action):
        self.cube.rotation_euler[2] = action

    def _env_post_step(self):
        self.count += 1
        angle = self.cube.rotation_euler[2]
        return dict(
            obs=angle,
            reward=1. if abs(angle) > 0.5 else 0.,
            done=self.events.frameid > self.done_after,
            count=self.count
        )

def main():    
    args, remainder = btb.parse_blendtorch_args()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--done-after', default=10, type=int)
    envargs = parser.parse_args(remainder)

    agent = btb.env.RemoteControlledAgent(
        args.btsockets['GYM']        
    )
    env = MyEnv(agent, done_after=envargs.done_after)
    if not bpy.app.background:
        env.attach_default_renderer(every_nth=1)
    env.run(frame_range=(1,10000), use_animation=not bpy.app.background)

main()

