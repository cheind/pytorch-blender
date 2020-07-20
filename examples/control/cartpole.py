import bpy
from mathutils import Matrix
import numpy as np

from blendtorch.btb import gym

class CartpoleEnv(gym.BaseEnv):
    def __init__(self, agent, frame_range=None):
        self.cart = bpy.data.objects['Cart']
        self.pole = bpy.data.objects['Pole']
        self.polerot = bpy.data.objects['PoleRotHelp']        
        self.motor = bpy.data.objects['Motor'].rigid_body_constraint
        self.fps = bpy.context.scene.render.fps
        super().__init__(agent, frame_range)

    def _env_reset(self, ctx):
        super()._env_reset(ctx)
        self.motor.motor_lin_target_velocity = 0.
        self.cart.location = (0.0,0,1.2)
        self.polerot.rotation_euler[1] = np.random.uniform(-0.6,0.6)

        
        # self.cart.matrix_world.translation = (0.0, 0, 1.2)
        # e = self.polerot.matrix_world.to_euler('XYZ')
        # e.y = np.random.uniform(-0.6,0.6)
        # m = e.to_matrix()
        # m.resize_4x4()
        # self.polerot.matrix_world = Matrix.Translation(self.polerot.matrix_world.translation) @ m
        

    def _env_prepare_step(self, action, ctx):
        self._apply_motor_force(action)
        
    def _env_post_step(self, ctx):
        c = self.cart.matrix_world.translation[0]
        p = self.pole.matrix_world.translation[0]
        a = self.polerot.matrix_world.to_euler('XYZ')[1]
        ctx.obs = (c,p,a)
        ctx.reward = 0.
        ctx.done = abs(a) > 0.6 or abs(c) > 3.0
        if ctx.action is None:
            print(c,p,a) # why 2 times after reset?

    def _apply_motor_force(self, f):
        # a = f/m
        # assuming constant acceleration between two steps we have
        # v_(t+1) = v(t) + a*dt, from which we get
        # v_(t+1) = v(t) + (f/m)*dt
        self.motor.motor_lin_target_velocity = self.motor.motor_lin_target_velocity + f/self.cart.rigid_body.mass/self.fps

def agent(ctx):
    if ctx.done:
        return None
    c,p,_ = ctx.obs
    return (p-c)*15

env = CartpoleEnv(agent, frame_range=(1,10000))



# cart = bpy.data.objects['Cart']
# pole = bpy.data.objects['Pole']
# rothelp = bpy.data.objects['PoleRotHelp']
# mass = cart.rigid_body.mass
# motor = bpy.data.objects['Motor'].rigid_body_constraint
# fps = bpy.context.scene.render.fps
# control = False

# class Controller:
#     def __init__(self, cart, pole):
#         self.cart = cart
#         self.pole = pole

#     def control(self):
#         c = self.cart.matrix_world.translation[0]
#         p = self.pole.matrix_world.translation[0]
#         e = (p-c)
#         return e*20

# ctrl = Controller(cart, pole)

# def apply_motor_force(f):
#     # a = f/m
#     # assuming constant acceleration between two steps we have
#     # v_(t+1) = v(t) + a*dt, from which we get
#     # v_(t+1) = v(t) + (f/m)*dt
#     motor.motor_lin_target_velocity = motor.motor_lin_target_velocity + f/mass/fps

# def before_animation():
#     cart.location = (0.0,0,1.2)
#     rothelp.rotation_euler[1] = np.random.uniform(-1.,1.)
#     #apply_motor_force(np.random.uniform(-10,10))

# def pre_frame(scene, *args):
#     f = ctrl.control()
#     apply_motor_force(f)

# def post_frame(scene, *args):
#     pass

# bpy.ops.screen.animation_cancel()
# bpy.context.scene.frame_set(1)
# bpy.app.handlers.frame_change_pre.append(pre_frame)
# bpy.app.handlers.frame_change_post.append(post_frame)
# before_animation()

