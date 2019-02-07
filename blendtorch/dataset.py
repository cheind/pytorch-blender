
class BlenderDataset:
    '''A dataset that reads from Blender publishers.'''

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


    def __getitem__(self, idx):
        d = self.blender_launcher.recv(timeoutms=self.timeoutms)        

        x, coords = d['image'], d['xy']
        
        h,w = x.shape[0], x.shape[1]
        coords[...,1] = 1. - coords[...,1] # Blender uses origin bottom-left.        
        coords *= np.array([w,h])[None, :]

        if self.transforms:
            x = self.transforms(x)
        return x, coords, d['id']

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    with BlenderLaunch(num_instances=4, instance_args=[['-id', '0'], ['-id', '1'], ['-id', '2'], ['-id', '3']]) as bl:
        ds = BlenderDataset(bl)
        # Note, in the following num_workers must be 0
        dl = data.DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)

        for idx in range(10):
            x, coords, ids = next(iter(dl))
            print(f'Received from {ids}')

            # Drawing is the slow part ...
            fig, axs = plt.subplots(2,2,frameon=False, figsize=(16*2,9*2))
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
            axs = axs.reshape(-1)
            for i in range(x.shape[0]):
                axs[i].imshow(x[i], aspect='auto', origin='upper')
                axs[i].scatter(coords[i][:, 0], coords[i][:, 1], s=100)
                axs[i].set_axis_off()
            fig.savefig(f'tmp/output_{idx}.png')
            plt.close(fig)
