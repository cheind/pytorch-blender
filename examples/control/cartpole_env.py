import bpy
from mathutils import Matrix
import numpy as np

from blendtorch import btb

class CartpoleEnv(btb.gym.BaseEnv):
    def __init__(self, agent, frame_range=None, use_animation=True, offline_render=True):
        self.cart = bpy.data.objects['Cart']
        self.pole = bpy.data.objects['Pole']
        self.polerot = bpy.data.objects['PoleRotHelp']        
        self.motor = bpy.data.objects['Motor'].rigid_body_constraint
        self.fps = bpy.context.scene.render.fps
        self.off = btb.OffScreenRenderer()
        self.off.view_matrix = btb.camera.view_matrix()
        self.off.proj_matrix = btb.camera.projection_matrix()
        super().__init__(
            agent, 
            frame_range=frame_range,
            use_animation=use_animation,
            offline_render=offline_render
        )

    def _env_reset(self, ctx):
        super()._env_reset(ctx)
        self.motor.motor_lin_target_velocity = 0.
        self.cart.location = (0.0,0,1.2)
        self.polerot.rotation_euler[1] = np.random.uniform(-0.6,0.6)

    def _env_prepare_step(self, action, ctx):
        self._apply_motor_force(action)
        
    def _env_post_step(self, ctx):
        c = self.cart.matrix_world.translation[0]
        p = self.pole.matrix_world.translation[0]
        a = self.polerot.matrix_world.to_euler('XYZ')[1]
        ctx.obs = (c,p,a,self.off.render())
        ctx.reward = 0.
        ctx.done = abs(a) > 0.6 or abs(c) > 3.0

    def _apply_motor_force(self, f):
        # a = f/m
        # assuming constant acceleration between two steps we have
        # v_(t+1) = v(t) + a*dt, from which we get
        # v_(t+1) = v(t) + (f/m)*dt
        self.motor.motor_lin_target_velocity = self.motor.motor_lin_target_velocity + f/self.cart.rigid_body.mass/self.fps

def main():
    args, remainder = btb.parse_blendtorch_args()
    agent = btb.gym.RemoteControlledAgent(args.btsockets['GYM'])
    env = CartpoleEnv(agent, frame_range=(1,10000), offline_render=True, use_animation=True)

main()

