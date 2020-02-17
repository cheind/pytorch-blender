
class Receiver:
    '''Base class for reading from blender instances'''

    def __init__(self, blender_launcher):
        '''Initialize instance
        
        Params
        ------
        blender_launcher: BlenderLauncher
            Active instance of BlenderLauncher
        '''

        self.blender_launcher = blender_launcher
  
    def recv(self, timeoutms=5000):
        '''Receive from Blender instances.'''
        return self.blender_launcher.recv(timeoutms) 

    def __call__(self, timeoutms=5000):
        return self.recv(timeoutms)
