from functools import partial

class Signal:
    def __init__(self):
        self.slots = []

    def add(self, fn, *args, **kwargs):
        fnp = partial(fn, *args, **kwargs)        
        self.slots.append(fnp)
        return fnp

    def remove(self, handle):
        self.slots.remove(handle)

    def __call__(self, *args, **kwargs):
        for s in self.slots:
            s(*args, **kwargs)


if __name__ == '__main__':
    s = Signal()

    def printme(arg, arg2, kw=None):
        print(arg, arg2, kw)

    s.add(printme, arg2=10, kw='hello')
    s.add(printme, arg2=5, kw='wuff')

    s('common')