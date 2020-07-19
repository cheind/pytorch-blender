
import bpy

from .signal import Signal

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

    def play(self, frame_range=None, repeat=-1):
        raise NotImplementedError()

    def _set_frame_range(self, frame_range):
        bpy.context.scene.frame_start = frame_range[0]
        bpy.context.scene.frame_end = frame_range[1]

    @property
    def frameid(self):
        return bpy.context.scene.frame_current

class SteppingAnimationController(AnimationControllerBase): 
    
    def play(self, once=False, startframe=None, stopframe=None):
        self._set_frame_range(startframe, stopframe)
        while True:
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
        self.playing = False
        
    def play(self, frame_range=None, repeat=-1):
        if self.playing:
            self.cancel()
        if frame_range is None:
            frame_range = (bpy.context.scene.frame_start, bpy.context.scene.endframe)
        self._set_frame_range(frame_range)
        self.frame_range = frame_range
        self.repeat_count = 0
        self.repeat_max = repeat
        self.pre_play.invoke()
        bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        bpy.app.handlers.frame_change_post.append(self._on_post_frame)
        bpy.context.scene.frame_set(frame_range[0])
        bpy.ops.screen.animation_play()
        self.playing = True        

    def _on_pre_frame(self, scene, *args):              
        pre_first = (self.frameid == self.frame_range[0])
        
        if pre_first:
            self.pre_animation.invoke()
        self.pre_frame.invoke()

    def _on_post_frame(self, scene, *args):
        post_last = (self.frameid == self.frame_range[1])

        self.post_frame.invoke()

        if post_last:
            self.repeat_count += 1
            self.post_animation.invoke()
            if self.repeat_count == self.repeat_max:
                self.cancel()

    def cancel(self):
        if not self.playing:
            return
        bpy.app.handlers.frame_change_pre.remove(self._on_pre_frame)
        bpy.app.handlers.frame_change_post.remove(self._on_post_frame)
        bpy.ops.screen.animation_cancel(restore_frame=False)
        self.playing = False
        self.post_play.invoke()