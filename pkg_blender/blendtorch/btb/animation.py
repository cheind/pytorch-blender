
import sys
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
        self.post_pixel = Signal()

    def play(self, frame_range=None, num_episodes=-1):
        raise NotImplementedError()

    def _set_frame_range(self, frame_range):
        bpy.context.scene.frame_start = frame_range[0]
        bpy.context.scene.frame_end = frame_range[1]

    @property
    def frameid(self):
        return bpy.context.scene.frame_current

class SteppingAnimationController(AnimationControllerBase): 
    
    def play(self, frame_range=None, num_episodes=-1):       
        if frame_range is None:
            frame_range = (bpy.context.scene.frame_start, bpy.context.scene.frame_end) 
        if num_episodes <= 0:
            num_episodes = sys.maxsize
        self._set_frame_range(frame_range)

        self.pre_play.invoke()
        for e in range(num_episodes):
            self.pre_animation.invoke()
            for i in range(frame_range[0], frame_range[1]+1):
                self.pre_frame.invoke()
                bpy.context.scene.frame_set(i)
                self.post_frame.invoke()
                self.post_pixel.invoke()
            self.post_animation.invoke()            
        self.post_play.invoke()

class AnimationController(AnimationControllerBase):    

    def __init__(self):    
        super().__init__()
        self.playing = False
        
    def play(self, frame_range=None, num_episodes=-1):
        if self.playing:
            self.cancel()
        if frame_range is None:
            frame_range = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
        self._set_frame_range(frame_range)
        self.frame_range = frame_range
        self.episode = 0
        self.num_episodes = num_episodes
        self.pre_play.invoke()
        bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        bpy.app.handlers.frame_change_post.append(self._on_post_frame)
        self.area,self.space,region = self.find_view3d()
        bpy.types.SpaceView3D.draw_handler_add(self._on_post_pixel, (), 'WINDOW', 'POST_PIXEL')        
        bpy.context.scene.frame_set(frame_range[0])
        bpy.ops.screen.animation_play()
        self.playing = True        

    def _on_pre_frame(self, scene, *args):              
        pre_first = (self.frameid == self.frame_range[0])
        
        if pre_first:
            self.pre_animation.invoke()
        self.pre_frame.invoke()

    def _on_post_frame(self, scene, *args):
        self.post_frame.invoke()

    def _on_post_pixel(self):
        if not bpy.context.space_data == self.space or not self.playing:
            return
        
        self.post_pixel.invoke()        

        post_last = (self.frameid == self.frame_range[1])
        if post_last:
            self.episode += 1
            self.post_animation.invoke()
            if self.episode == self.num_episodes:
                self.cancel()

    def cancel(self):
        if not self.playing:
            return
        bpy.app.handlers.frame_change_pre.remove(self._on_pre_frame)
        bpy.app.handlers.frame_change_post.remove(self._on_post_frame)
        bpy.ops.screen.animation_cancel(restore_frame=False)
        self.playing = False
        self.post_play.invoke()

    def find_view3d(self):
        areas = [a for a in bpy.context.screen.areas if a.type == 'VIEW_3D']
        assert len(areas) > 0
        area = areas[0]
        window_region = sorted([r for r in area.regions if r.type == 'WINDOW'], key=lambda x:x.width, reverse=True)[0]        
        spaces = [s for s in areas[0].spaces if s.type == 'VIEW_3D']
        assert len(spaces) > 0
        return area, spaces[0], window_region