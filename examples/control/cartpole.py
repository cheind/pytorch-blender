import bpy
import numpy as np

cart = bpy.data.objects['Cart']
pole = bpy.data.objects['Pole']
rothelp = bpy.data.objects['PoleRotHelp']
mass = cart.rigid_body.mass
motor = bpy.data.objects['Motor'].rigid_body_constraint
fps = bpy.context.scene.render.fps
control = False

class Controller:
    def __init__(self, cart, pole):
        self.cart = cart
        self.pole = pole

    def control(self):
        c = self.cart.matrix_world.translation[0]
        p = self.pole.matrix_world.translation[0]
        e = (p-c)
        return e*20

ctrl = Controller(cart, pole)

def apply_motor_force(f):
    # a = f/m
    # assuming constant acceleration between two steps we have
    # v_(t+1) = v(t) + a*dt, from which we get
    # v_(t+1) = v(t) + (f/m)*dt
    motor.motor_lin_target_velocity = motor.motor_lin_target_velocity + f/mass/fps

def before_animation():
    cart.location = (0.0,0,1.2)
    rothelp.rotation_euler[1] = np.random.uniform(-1.,1.)
    #apply_motor_force(np.random.uniform(-10,10))

def pre_frame(scene, *args):
    f = ctrl.control()
    apply_motor_force(f)

def post_frame(scene, *args):
    pass

bpy.ops.screen.animation_cancel()
bpy.context.scene.frame_set(1)
bpy.app.handlers.frame_change_pre.append(pre_frame)
bpy.app.handlers.frame_change_post.append(post_frame)
before_animation()

