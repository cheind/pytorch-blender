import bpy
from mathutils import Matrix
import numpy as np

from blendtorch import btb


class CartpoleEnv(btb.env.BaseEnv):
    def __init__(self, agent):
        super().__init__(agent)
        self.cart = bpy.data.objects['Cart']
        self.pole = bpy.data.objects['Pole']
        self.polerot = bpy.data.objects['PoleRotHelp']
        self.motor = bpy.data.objects['Motor'].rigid_body_constraint
        # Note, ensure that physics run at same speed.
        self.fps = bpy.context.scene.render.fps
        self.total_mass = self.cart.rigid_body.mass + self.pole.rigid_body.mass

    def _env_reset(self):
        c = 0.
        p = np.array([0.0, 0, 1.2])
        a = np.random.uniform(-0.6, 0.6)
        self.motor.motor_lin_target_velocity = c
        self.cart.location = p
        self.polerot.rotation_euler[1] = a
        return dict(
            obs=(c, p, a),
            reward=0.,
            done=False
        )

    def _env_prepare_step(self, action):
        self._apply_motor_force(action)

    def _env_post_step(self):
        c = self.cart.matrix_world.translation[0]
        p = self.pole.matrix_world.translation[0]
        a = self.pole.matrix_world.to_euler('XYZ')[1]
        return dict(
            obs=(c, p, a),
            reward=0.,
            done=bool(
                abs(a) > 0.6
                or abs(c) > 4.0
            )
        )

    def _apply_motor_force(self, f):
        # a = f/m
        # assuming constant acceleration between two steps we have
        # v_(t+1) = v(t) + a*dt, from which we get
        # v_(t+1) = v(t) + (f/m)*dt
        self.motor.motor_lin_target_velocity = self.motor.motor_lin_target_velocity + \
            f/self.total_mass/self.fps


def main():
    args, remainder = btb.parse_blendtorch_args()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--render-every', default=None, type=int)
    parser.add_argument('--real-time', dest='realtime', action='store_true')
    parser.add_argument('--no-real-time', dest='realtime',
                        action='store_false')
    envargs = parser.parse_args(remainder)

    agent = btb.env.RemoteControlledAgent(
        args.btsockets['GYM'],
        real_time=envargs.realtime
    )
    env = CartpoleEnv(agent)
    env.attach_default_renderer(every_nth=envargs.render_every)
    env.run(frame_range=(1, 10000), use_animation=True)


main()
