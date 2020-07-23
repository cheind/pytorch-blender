
import sys
import bpy
import threading

from .signal import Signal
from .utils import find_first_view3d

class AnimationController:  
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
        self._plyctx = None

    class _PlayContext:
        def __init__(self, frame_range, num_episodes, use_animation, use_offline_render):
            self.frame_range = frame_range
            self.use_animation = use_animation
            self.use_offline_render = use_offline_render
            self.episode = 0
            self.num_episodes = num_episodes
            self.pending_post_pixel = False
            self.draw_handler = None
            self.draw_space = None

        def skip_post_frame(self):
            return (
                self.use_offline_render and 
                self.use_animation and
                (
                    bpy.context.space_data != self.draw_space or
                    not self.pending_post_pixel
                )
            )


    @property
    def frameid(self):
        return bpy.context.scene.frame_current
        
    def play(self, frame_range=None, num_episodes=-1, use_animation=True, use_offline_render=True, use_physics=True):
        assert self._plyctx is None, 'Animation already running'

        self._plyctx = AnimationController._PlayContext(
            frame_range=AnimationController.setup_frame_range(frame_range, physics=use_physics),
            num_episodes=(num_episodes if num_episodes >= 0 else sys.maxsize),
            use_animation=use_animation,
            use_offline_render=use_offline_render
        )

        if use_animation:
            self._play_animation()
        else:
            self._play_manual()

    @staticmethod
    def setup_frame_range(frame_range, physics=True):
        if frame_range is None:
            frame_range = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
        bpy.context.scene.frame_start = frame_range[0]
        bpy.context.scene.frame_end = frame_range[1]
        if physics and bpy.context.scene.rigidbody_world:
            bpy.context.scene.rigidbody_world.point_cache.frame_start = frame_range[0]
            bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_range[1]
        return frame_range

    def _play_animation(self):
        self.pre_play.invoke()     
        bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        if self._plyctx.use_offline_render:    
            _, self._plyctx.draw_space, _ = find_first_view3d()
            self._plyctx.draw_handler = bpy.types.SpaceView3D.draw_handler_add(self._on_post_frame, (), 'WINDOW', 'POST_PIXEL')
        else:
            bpy.app.handlers.frame_change_post.append(self._on_post_frame)
        bpy.context.scene.frame_set(self._plyctx.frame_range[0])
        bpy.ops.screen.animation_play()

    def _play_manual(self):
        self.pre_play.invoke()
        bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        bpy.app.handlers.frame_change_post.append(self._on_post_frame)

        while self._plyctx.episode  < self._plyctx.num_episodes:
            bpy.context.scene.frame_set(self._plyctx.frame_range[0])
            while self.frameid < self._plyctx.frame_range[1]:
                bpy.context.scene.frame_set(self.frameid+1)

    def rewind(self):
        if self._plyctx is not None:
            self._set_frame(self._plyctx.frame_range[0])

    def _set_frame(self, frame_index):
        bpy.context.scene.frame_set(frame_index)

    def _on_pre_frame(self, scene, *args):              
        pre_first = (self.frameid == self._plyctx.frame_range[0])
        
        if pre_first:
            self.pre_animation.invoke()
        self.pre_frame.invoke()
        self._plyctx.pending_post_pixel = True

    def _on_post_frame(self, *args):
        if self._plyctx.skip_post_frame():
            return
        self._plyctx.pending_post_pixel = False
        
        self.post_frame.invoke()
        post_last = (self.frameid == self._plyctx.frame_range[1])
        if post_last:            
            self.post_animation.invoke()
            self._plyctx.episode += 1
            if self._plyctx.episode == self._plyctx.num_episodes:
                self._cancel()

    def _cancel(self):
        bpy.app.handlers.frame_change_pre.remove(self._on_pre_frame)
        bpy.app.handlers.frame_change_post.remove(self._on_post_frame)
        if self._plyctx.draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._plyctx.draw_handler)
            self._plyctx.draw_handler = None
        bpy.ops.screen.animation_cancel(restore_frame=False)        
        self.post_play.invoke()
        del self._plyctx
        self._plyctx = None

    