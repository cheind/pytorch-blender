
import bpy

from .signal import Signal

class Controller:
    '''Animation controller with fine-grained callbacks.'''
    
    def __init__(self):
        self.pre_animation = Signal()
        self.pre_frame = Signal()
        self.post_animation = Signal()
        self.is_playing = False
        self.h_pre_frame = bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        
    def play(self, once=True, startframe=None, endframe=None):
        self._set_frame_range(startframe, endframe)
        self.once = once
        self.is_playing = True        
        bpy.ops.screen.animation_play()       

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
        post_last = (bpy.context.scene.frame_current == bpy.context.scene.frame_end + 1)

        if self.once and post_last:
            bpy.ops.screen.animation_cancel()
            self.is_playing = False
            self.post_animation.invoke()
        else:
            if pre_first:
                self.pre_animation.invoke()
            self.pre_frame.invoke()