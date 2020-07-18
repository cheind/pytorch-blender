
import bpy

from .signal import Signal

class AnimationControllerBase:
    '''Animation controller base class with fine-grained callbacks.
    
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

    def play(self, once=False, startframe=None, stopframe=None):
        raise NotImplementedError()

    def _set_frame_range(self, startframe, endframe):
        startframe = startframe or bpy.context.scene.frame_start
        endframe = endframe or bpy.context.scene.endframe
        bpy.context.scene.frame_start = startframe
        bpy.context.scene.frame_end = endframe
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)

class SteppingAnimationController(AnimationControllerBase): 
    
    def play(self, once=False, startframe=None, stopframe=None):
        self._set_frame_range(startframe, stopframe)
        while True:
            print('start')
            self.pre_animation.invoke()
            for i in range(startframe, stopframe+1):
                self.pre_frame.invoke()
                bpy.context.scene.frame_set(i)
                self.post_frame.invoke()
            self.post_animation.invoke()
            if once:
                break

class AnimationController(AnimationControllerBase):   
    def __init__(self):    
        super().__init__()    
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