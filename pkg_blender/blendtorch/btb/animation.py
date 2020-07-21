
import sys
import bpy
import threading

from .signal import Signal
from .utils import find_first_view3d

class AnimationControllerBase:
    '''Animation controller base class with fine-grained callbacks.
    
    Exposes the following signals
     - pre_play() invoked before playing starts
     - pre_animation() invoked before first frame of animation range is processed
     - pre_frame() invoked before a frame begins
     - post_frame() invoked after a frame is finished
     - post_animation() invoked after the last animation frame has completed
     - post_play() invoked after playing ends
    '''

    def __init__(self):
        self.pre_animation = Signal()
        self.pre_frame = Signal()
        self.post_frame = Signal()
        self.post_animation = Signal()
        self.pre_play = Signal()
        self.post_play = Signal()

    def play(self, frame_range=None, num_episodes=-1):
        raise NotImplementedError()

    def _set_frame_range(self, frame_range):
        bpy.context.scene.frame_start = frame_range[0]
        bpy.context.scene.frame_end = frame_range[1]

    @property
    def frameid(self):
        return bpy.context.scene.frame_current

class AnimationController(AnimationControllerBase):    

    def __init__(self):    
        super().__init__()   

    def _setup_frame_range(self, frame_range):
        if frame_range is None:
            frame_range = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
        bpy.context.scene.frame_start = frame_range[0]
        bpy.context.scene.frame_end = frame_range[1]
        return frame_range

        
    def play(self, frame_range=None, num_episodes=-1, use_animation=True, offline_render=True):
        self.frame_range = self._setup_frame_range(frame_range)
        self.use_animation = use_animation
        self.require_offline_render = offline_render
        self.episode = 0
        self.num_episodes = num_episodes if num_episodes >= 0 else sys.maxsize
        self._pending_post_pixel = False
        self._draw_handler = None

        if use_animation:
            self._play_animation()
        else:
            self._play_manual()

    def _play_animation(self):
        self.pre_play.invoke()     
        bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        if self.require_offline_render:    
            self.area, self.space, self.region = find_first_view3d()
            self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(self._on_post_frame, (), 'WINDOW', 'POST_PIXEL')
        else:
            bpy.app.handlers.frame_change_post.append(self._on_post_frame)
        bpy.context.scene.frame_set(self.frame_range[0])
        bpy.ops.screen.animation_play()

    def _play_manual(self):
        self.pre_play.invoke()
        bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        bpy.app.handlers.frame_change_post.append(self._on_post_frame)

        while self.episode  < self.num_episodes:
            bpy.context.scene.frame_set(self.frame_range[0])
            while self.frameid < self.frame_range[1]:
                bpy.context.scene.frame_set(self.frameid+1)
            self.episode += 1

    def reset(self):
        self.set_frame(self.frame_range[0])

    def set_frame(self, frame_index):
        bpy.context.scene.frame_set(frame_index)

    def step(self):
        bpy.context.scene.frame_set(self.frameid + 1)

    def _on_pre_frame(self, scene, *args):              
        pre_first = (self.frameid == self.frame_range[0])
        
        if pre_first:
            self.pre_animation.invoke()
        self.pre_frame.invoke()
        self._pending_post_pixel = True

    def _on_post_frame(self, *args):
        if (self.require_offline_render and 
                self.use_animation and
                (bpy.context.space_data != self.space or 
                    not self._pending_post_pixel)):
            return
        self._pending_post_pixel = False
        
        self.post_frame.invoke()
        post_last = (self.frameid == self.frame_range[1])
        if post_last:
            self.episode += 1
            self.post_animation.invoke()
            if self.episode == self.num_episodes:
                self._cancel()

    def _cancel(self):
        bpy.app.handlers.frame_change_pre.remove(self._on_pre_frame)
        bpy.app.handlers.frame_change_post.remove(self._on_post_frame)
        if self._draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler)
            self._draw_handler = None
        bpy.ops.screen.animation_cancel(restore_frame=False)
        self.post_play.invoke()

# ctrl = AnimationController(rendercap=False)
# ctrl.auto_play(framerange, num_episodes)
# ctrl.