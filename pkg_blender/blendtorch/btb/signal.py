from functools import partial

class Signal:
    '''Distribute messages to multiple callbacks.'''
    def __init__(self):
        self.slots = []

    def add(self, fn, *args, **kwargs):
        '''Register `fn` as callback.
        
        Params
        ------
        *args: optional
            Additional positional arguments to provide callback with
        **kwargs: optional
            Additional keyword arguments to provide callback with

        Returns
        -------
        handle: object
            Handle that can be used to unregister callback.
        '''
        fnp = partial(fn, *args, **kwargs)        
        self.slots.append(fnp)
        return fnp

    def remove(self, handle):
        '''Unregister callback using handle returned from `add`.'''
        self.slots.remove(handle)

    def invoke(self, *args, **kwargs):
        '''Invoke the signal.

        Params
        ------
        *args: optional
            Positional arguments to send to all callbacks
        **kwargs: optional
            Keyword arguments to send to all callbacks
        '''
        for s in self.slots:
            s(*args, **kwargs)