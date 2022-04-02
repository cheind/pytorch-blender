import sys
import bpy
from contextlib import contextmanager

from .signal import Signal
from .utils import find_first_view3d


class AnimationController:
    """Provides an interface to Blender's animation system with fine-grained callbacks.

    To play nice with Blender, blendtorch provides a callback based class for interacting
    with the Blender's animation and rendering system. The usual way to interact with
    this class is through an object of AnimationController. Depending on the required
    callbacks, one or more signals are connected to Python functions.
    `AnimationController.play` starts the animation loop.

    By default `AnimationController.play` is non-blocking and therefore requires a
    non background instance of Blender. In case `--background` is required,
    `AnimationController.play` also supports blocking animation loop variant. In blocking
    execution, offscreen rendering works but may crash Blender once the loop is exited (2.83.2),
    and is therefore not recommended when image data is required.

    `AnimationController` exposes the following signals
     - pre_play() invoked before playing starts
     - pre_animation() invoked before first frame of animation range is processed
     - pre_frame() invoked before a frame begins
     - post_frame() invoked after a frame is finished
     - post_animation() invoked after the last animation frame has completed
     - post_play() invoked after playing ends
    """

    def __init__(self):
        """Create a new instance."""
        self.pre_animation = Signal()
        self.pre_frame = Signal()
        self.post_frame = Signal()
        self.post_animation = Signal()
        self.pre_play = Signal()
        self.post_play = Signal()
        self._plyctx = None

    class _PlayContext:
        """Internal bookkeeping of animation veriables."""

        def __init__(
            self, frame_range, num_episodes, use_animation, use_offline_render
        ):
            self.frame_range = frame_range
            self.use_animation = use_animation
            self.use_offline_render = use_offline_render
            self.episode = 0
            self.num_episodes = num_episodes
            self.pending_post_frame = False
            self.draw_handler = None
            self.draw_space = None
            self.last_post_frame = 0
            self._allow_events = True

        def skip_post_frame(self, current_frame):
            return (
                not self.allow_events
                or not self.pending_post_frame
                or self.last_post_frame == current_frame
                or (
                    self.use_animation
                    and self.use_offline_render
                    and bpy.context.space_data != self.draw_space
                )
            )

        @contextmanager
        def disable_events(self):
            old = self._allow_events
            self._allow_events = False
            yield
            self._allow_events = old

        @contextmanager
        def enable_events(self):
            old = self._allow_events
            self._allow_events = True
            yield
            self._allow_events = old

        @property
        def allow_events(self):
            return self._allow_events

    @property
    def frameid(self):
        """Returns the current frame id."""
        return bpy.context.scene.frame_current

    def play(
        self,
        frame_range=None,
        num_episodes=-1,
        use_animation=True,
        use_offline_render=True,
        use_physics=True,
    ):
        """Start the animation loop.

        Params
        ------
        frame_range: tuple
            Start and end of frame range to play. Note that start and end are inclusive.
        num_episodes: int
            The number of loops to play. -1 loops forever.
        use_animation: bool
            Whether to use Blender's non-blocking animation system or use a blocking variant.
            By default True. When True, allows BlenderUI to refresh and be responsive. The animation
            will be run in target FPS. When false, does not allow Blender UI to refresh. The animation
            runs as fast as it can.
        use_offline_render: bool
            Whether offline rendering should be supported. By default True. When True, calls to
            `OffscreenRenderer` are safe inside the `post_frame` callback.
        use_physics: bool
            Whether physics should be enabled. Default is True. When True, sets the simulation range
            to match the animation frame range.
        """
        assert self._plyctx is None, "Animation already running"

        self._plyctx = AnimationController._PlayContext(
            frame_range=AnimationController.setup_frame_range(
                frame_range, physics=use_physics
            ),
            num_episodes=(num_episodes if num_episodes >= 0 else sys.maxsize),
            use_animation=use_animation,
            use_offline_render=use_offline_render,
        )

        if use_animation:
            self._play_animation()
        else:
            self._play_manual()

    @staticmethod
    def setup_frame_range(frame_range, physics=True):
        """Setup the animation and physics frame range.

        Params
        ------
        frame_range: tuple
            Start and end (inclusive) frame range to be animated.
            Can be None, in which case the scenes frame range is used.
        physics: bool
            Whether or not to apply the frame range settings to the rigid body
            simulation.

        Returns
        -------
        frame_range: tuple
            the updated frame range.
        """

        if frame_range is None:
            frame_range = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)
        bpy.context.scene.frame_start = frame_range[0]
        bpy.context.scene.frame_end = frame_range[1]
        if physics and bpy.context.scene.rigidbody_world:
            bpy.context.scene.rigidbody_world.point_cache.frame_start = frame_range[0]
            bpy.context.scene.rigidbody_world.point_cache.frame_end = frame_range[1]
        return frame_range

    def _play_animation(self):
        """Setup and start Blender animation loop."""
        with self._plyctx.disable_events():
            self.pre_play.invoke()
        bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        if self._plyctx.use_offline_render:
            # To be save, we need to draw from `POST_PIXEL` not `frame_change_post`.
            # However `POST_PIXEL` might be called more than once per frame. We therefore
            # set and release `pending_post_pixel` to match things up.
            _, self._plyctx.draw_space, _ = find_first_view3d()
            self._plyctx.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                self._on_post_frame, (), "WINDOW", "POST_PIXEL"
            )
        else:
            bpy.app.handlers.frame_change_post.append(self._on_post_frame)
        # Set to first frame.
        bpy.context.scene.frame_set(self._plyctx.frame_range[0])
        # The following does not block. Note, in --offscreen this does nothing.
        bpy.ops.screen.animation_play()

    def _play_manual(self):
        """Setup and start blocking animation loop."""
        with self._plyctx.disable_events():
            self.pre_play.invoke()
        bpy.app.handlers.frame_change_pre.append(self._on_pre_frame)
        bpy.app.handlers.frame_change_post.append(self._on_post_frame)

        while self._plyctx.episode < self._plyctx.num_episodes:
            bpy.context.scene.frame_set(self._plyctx.frame_range[0])
            while self.frameid < self._plyctx.frame_range[1]:
                bpy.context.scene.frame_set(self.frameid + 1)
                if (
                    self._plyctx is None
                ):  # The above frame_set might have called _cancel,
                    return  # which in turn deletes _plyctx

    def rewind(self):
        """Request resetting the animation to first frame."""
        if self._plyctx is not None:
            self._set_frame(self._plyctx.frame_range[0])

    def _set_frame(self, frame_index):
        """Step to a specific frame."""
        with self._plyctx.enable_events():  # needed for env support?
            bpy.context.scene.frame_set(frame_index)

    def _on_pre_frame(self, scene, *args):
        """Handle pre-frame events internally."""
        if not self._plyctx.allow_events:
            return

        pre_first = self.frameid == self._plyctx.frame_range[0]

        with self._plyctx.disable_events():
            if pre_first:
                self.pre_animation.invoke()
            self.pre_frame.invoke()
        # The following guards us from multiple calls to `_on_post_frame`
        # when we hooked into `POST_PIXEL`
        self._plyctx.pending_post_frame = True

    def _on_post_frame(self, *args):
        """Handle post-frame events internally."""
        if self._plyctx.skip_post_frame(self.frameid):
            return
        self._plyctx.pending_post_frame = False
        self._plyctx.last_post_frame = self.frameid

        with self._plyctx.disable_events():
            self.post_frame.invoke()
            post_last = self.frameid == self._plyctx.frame_range[1]
            if post_last:
                self.post_animation.invoke()
                self._plyctx.episode += 1
                if self._plyctx.episode == self._plyctx.num_episodes:
                    self._cancel()

    def _cancel(self):
        """Stop the animation."""
        bpy.app.handlers.frame_change_pre.remove(self._on_pre_frame)
        if self._plyctx.draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(
                self._plyctx.draw_handler, "WINDOW"
            )
            self._plyctx.draw_handler = None
        else:
            bpy.app.handlers.frame_change_post.remove(self._on_post_frame)
        bpy.ops.screen.animation_cancel(restore_frame=False)
        self.post_play.invoke()
        del self._plyctx
        self._plyctx = None
