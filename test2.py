import bpy
import time

#bpy.context.scene.frame_set(1)
# import time
# t = time.time()
# for i in range(1,200):
#     bpy.context.scene.frame_set(i)

# print(bpy.data.objects['Cube1'].matrix_world.translation, time.time() - t)

from blendtorch import btb

off = btb.OffScreenRenderer()
off.view_matrix = btb.camera.view_matrix()
off.proj_matrix = btb.camera.projection_matrix()
#off.enabled = True

# from PIL import Image

# def post_frame():
#     print('post', bpy.data.objects['Cube1'].matrix_world.translation, bpy.context.scene.frame_current)
#     #off.area.tag_redraw()
#     data = off.render()
#     Image.fromarray(data).save(f'image_{anim.frameid}.png')

# t = time.time()
# anim = btb.SteppingAnimationController()
# anim.post_frame.add(post_frame)
# anim.play(once=True, startframe=1, stopframe=200)
# print(time.time() - t)






# ensure dynamics of rigid body are turned off
# ensure simulation runs long enough (stopframe)
# ensure same frame rate (physics solver and fps!)
# seems like the position corresponds to the mid-time between frames.
# 
#         eq.                          True             Blender                     
# 10 + -9.81*0.5*(100.5/60)**2 - > -3.761590625          -3.7612
# 10 + -9.81*0.5*(500.5/60)**2 - > -331.306590625      -331.3070


#https://blender.stackexchange.com/questions/156503/rendering-on-command-line-with-gpu




# from PIL import Image

# def pre_frame(anim):
#     print('pre-frame', anim.frameid)

# def post_frame(anim, off):
#     print('post-frame', anim.frameid)

# def post_pixel(anim, off):
#     print('post-pixel', anim.frameid)
#     data = off.render()[...,:3]
#     # print(data.shape, data.min(), data.max())
#     # Image.fromarray(data).save(f'image_{anim.frameid}.png')

# def pre_anim(anim):
#     print('pre-anim', anim.frameid)

# def post_anim(anim):
#     print('post-anim', anim.frameid)
#     print('----------')

# t = None

# def pre_play():
#     global t
#     t = time.time()
#     print('pre-play')
#     print('#########')


# def post_play():

#     print('#########')
#     print('post-play')        
#     print(time.time() - t)

# off = btb.OffScreenRenderer()
# off.view_matrix = btb.camera.view_matrix()
# off.proj_matrix = btb.camera.projection_matrix()

# anim = btb.AnimationController()
# anim.pre_play.add(pre_play)
# anim.post_play.add(post_play)
# anim.pre_animation.add(pre_anim, anim=anim)
# anim.post_animation.add(post_anim, anim=anim)
# anim.pre_frame.add(pre_frame, anim=anim)
# anim.post_frame.add(post_frame, anim=anim, off=off)
# anim.post_pixel.add(post_pixel, anim=anim, off=off)
# anim.play(frame_range=(1,60), repeat=1)



# # def post_image(img):
# #     print('post-image')

# # def pre_frame(*args):
# #     print('pre-frame')

# # from PIL import Image

# # def post_frame(*args):
# #     print('post-frame')
# #     data = off.render()
# #     img = Image.fromarray(data, 'RGBA').convert('RGB')
# #     img.save('img.png')


# # off.post_image.add(post_image)
# # #off.enabled = True
# # #
# # bpy.app.handlers.frame_change_pre.append(pre_frame)
# # bpy.app.handlers.frame_change_post.append(post_frame)


# # import time
# # t = time.time()
# # print('before_set')
# # for i in range(20):
# #     bpy.context.scene.frame_set(i)
# # print('set')
# # print(time.time() - t)


# # # pre-anim 1
# # # pre-frame 1
# # # post-frame 1


env = btb.gym.BaseEnv()
env.reset()


# import zmq

# #  Prepare our context and sockets
# context = zmq.Context()
# socket = context.socket(zmq.REQ)
# socket.connect("tcp://localhost:5559")

# #  Do 10 requests, waiting each time for a response
# for request in range(1,11):
#     socket.send(b"Hello")
#     message = socket.recv()
#     print("Received reply %s [%s]" % (request, message))