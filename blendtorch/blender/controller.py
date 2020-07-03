
import bpy

from .signal import Signal

class Controller:
    '''Animation controller with fine-grained callbacks.'''
    
    def __init__(self):
        self.before_animation = Signal()
        self.before_frame = Signal()
        self.after_animation = Signal()
        self.h_pre_frame = bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        
    def play(self, once=True, startframe=None, endframe=None):
        self._set_frame_range(startframe, endframe)
        self.once = once
        self.before_animation()
        bpy.ops.screen.animation_play()       

    def _set_frame_range(self, startframe, endframe):
        startframe = startframe or bpy.context.scene.frame_start
        endframe = endframe or bpy.context.scene.endframe
        bpy.context.scene.frame_start = startframe
        bpy.context.scene.frame_end = endframe
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)
                        
    def _on_pre_frame(self, scene, *args):
        if self.once and bpy.context.scene.frame_current == bpy.context.scene.frame_end:
            bpy.ops.screen.animation_cancel()
            self.after_animation()
        else:
            self.before_frame()