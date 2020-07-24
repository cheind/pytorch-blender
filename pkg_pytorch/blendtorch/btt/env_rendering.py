
RENDER_BACKENDS={}
LOOKUP_ORDER = ['openai', 'matplotlib']

def create_renderer(backend=None, **kwargs):
    if backend is None:
        avail = [RENDER_BACKENDS[l] for l in LOOKUP_ORDER if l in RENDER_BACKENDS]
        assert len(avail) > 0, 'No render backends available.'
        kls = avail[0]
    else:
        assert backend in RENDER_BACKENDS, f'Render backend {backend} not found.'
        kls = RENDER_BACKENDS[backend]
    return kls(**kwargs)

## MATPLOTLIB
try:
    import matplotlib.pyplot as plt

    class MatplotlibRenderer:
        def __init__(self, **kwargs):
            self.fig, self.ax = plt.subplots(1,1)
            self.img = None
        
        def imshow(self, rgb):
            if self.img is None:
                self.img = self.ax.imshow(rgb)
                plt.show(block=False)
                self.fig.canvas.draw()
            else:
                self.img.set_data(rgb)
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

        def close(self):
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None

        def __del__(self):
            self.close()
    
    RENDER_BACKENDS['matplotlib'] = MatplotlibRenderer
except ImportError as e:
    pass

## PYGLET/OpenAI based
try:
    from gym.envs.classic_control import rendering
    
    class OpenAIGymRenderer(object):

        def __init__(self, **kwargs):
            self._viewer = rendering.SimpleImageViewer(**kwargs)

        def imshow(self, rgb):
            self._viewer.imshow(rgb)

        def close(self):
            if self._viewer:
                self._viewer.close()
                self._viewer = None

        def __del__(self):
            self.close()

    RENDER_BACKENDS['openai'] = OpenAIGymRenderer
except ImportError as e:
    pass

