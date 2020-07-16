
import bpy

from .signal import Signal

class AnimationController:
    '''Animation controller with fine-grained callbacks.
    
    Exposes the following signals
     - pre_animation() invoked before first frame of animation range is processed
     - pre_frame() invoked before a frame begins
     - post_frame() invoked after a frame is finished
     - post_animation() invoked after the last animation frame has completed
    '''
    
    def __init__(self):
        self.pre_animation = Signal()
        self.pre_frame = Signal()
        self.post_frame = Signal()
        self.post_animation = Signal()
        self.is_playing = False
        self.h_pre_frame = bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        self.h_post_frame = bpy.app.handlers.frame_change_post.append(self._on_post_frame)
        
    def play(self, once=False, startframe=None, endframe=None):
        self._set_frame_range(startframe, endframe)
        self.once = once
        self.is_playing = True        
        self._on_pre_frame(bpy.context.scene)
        bpy.ops.screen.animation_play()

    def cancel(self):
        self.is_playing = False
        bpy.ops.screen.animation_cancel()

    def _set_frame_range(self, startframe, endframe):
        startframe = startframe or bpy.context.scene.frame_start
        endframe = endframe or bpy.context.scene.endframe
        bpy.context.scene.frame_start = startframe
        bpy.context.scene.frame_end = endframe
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)
                        
    def _on_pre_frame(self, scene, *args):  
        if not self.is_playing:
            return
        
        cur = bpy.context.scene.frame_current
        pre_first = (cur == bpy.context.scene.frame_start)
        
        if pre_first:
            self.pre_animation.invoke()
        self.pre_frame.invoke()

    def _on_post_frame(self, scene, *args):
        if not self.is_playing:
            return

        cur = bpy.context.scene.frame_current
        post_last = (cur == bpy.context.scene.frame_end)

        self.post_frame.invoke()

        if post_last:
            if self.once:
                self.is_playing = False
                bpy.ops.screen.animation_cancel(restore_frame=False)                
            self.post_animation.invoke()        