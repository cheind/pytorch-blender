
import bpy

from .signal import Signal

class Controller:
    '''Animation controller with fine-grained callbacks.'''
    
    def __init__(self):
        self.before_animation = Signal()
        self.before_frame = Signal()
        self.after_animation = Signal()
        self.h_pre_frame = bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        
    def play_once(self):
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)
        self.before_animation()
        bpy.ops.screen.animation_play()        
                        
    def _on_pre_frame(self, scene, *args):
        if bpy.context.scene.frame_current == bpy.context.scene.frame_end:
            bpy.ops.screen.animation_cancel()
            self.after_animation()
        else:
            self.before_frame()