import bpy
from mathutils import Matrix
import numpy as np

from blendtorch import btb

class CartpoleEnv(btb.gym.BaseEnv):
    def __init__(self, agent, frame_range=None, use_animation=True, render_every=None):
        self.cart = bpy.data.objects['Cart']
        self.pole = bpy.data.objects['Pole']
        self.polerot = bpy.data.objects['PoleRotHelp']
        self.motor = bpy.data.objects['Motor'].rigid_body_constraint
        self.fps = bpy.context.scene.render.fps # Note, ensure that physics run at same speed.
        self.total_mass = self.cart.rigid_body.mass + self.pole.rigid_body.mass
        self.render_every = render_every
        if self.render_every:
            self.off = btb.OffScreenRenderer(mode='rgb', gamma_coeff=2.2)
            self.off.view_matrix = btb.camera.view_matrix()
            self.off.proj_matrix = btb.camera.projection_matrix()
        else:
            self.off = None
        super().__init__(
            agent, 
            frame_range=frame_range,
            use_animation=use_animation,
            use_offline_render=self.render_every,
        )

    def _env_reset(self):
        self.motor.motor_lin_target_velocity = 0.
        self.cart.location = (0.0,0,1.2)
        self.polerot.rotation_euler[1] = np.random.uniform(-0.6,0.6)

    def _env_prepare_step(self, action):
        self._apply_motor_force(action)
        
    def _env_post_step(self):
        c = self.cart.matrix_world.translation[0]
        p = self.pole.matrix_world.translation[0]
        a = self.pole.matrix_world.to_euler('XYZ')[1]
        return dict(
            obs=(c,p,a),
            reward=0.,
            done=bool(
                abs(a) > 0.6 
                or abs(c) > 4.0
            ),
            rgb_array=self._render()
        )

    def _render(self):
        if self.off and (self.events.frameid % self.render_every) == 0:
            return self.off.render()
        else:
            return None

    def _apply_motor_force(self, f):
        # a = f/m
        # assuming constant acceleration between two steps we have
        # v_(t+1) = v(t) + a*dt, from which we get
        # v_(t+1) = v(t) + (f/m)*dt
        self.motor.motor_lin_target_velocity = self.motor.motor_lin_target_velocity + f/self.total_mass/self.fps

def main():    
    args, remainder = btb.parse_blendtorch_args()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--render_every', default=None, type=int)
    envargs = parser.parse_args(remainder)

    agent = btb.gym.RemoteControlledAgent(args.btsockets['GYM'])
    env = CartpoleEnv(agent, frame_range=(1,100), use_animation=True, render_every=envargs.render_every)

main()

