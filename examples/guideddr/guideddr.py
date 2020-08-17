'''Demonstrates adapting simulation parameters to match an empirical target distribution.
'''

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.distributions import LogNormal
import torchvision.utils as vutils

from blendtorch import btt

'''Batch size'''
BATCH = 64
'''Target label at descriminator'''
TARGET_LABEL = 1
'''Simulation label at descriminator'''
SIM_LABEL = 0
'''Number of Blender instances.'''
SIM_INSTANCES = 4
'''Long./Lat. log-normal supershape frequency (m1,m2) target mean'''
MEAN_TARGET = 2.25 # rough sample range: 8.5-10.5
'''Long./Lat. log-normal supershape frequency (m1,m2) target standard deviation'''
STD_TARGET = 0.1 

def generate_samples(n, mean, std):
    '''Generate supershape examples via parameter sampling.
    
    We assume all parameter except for m1/m2 to be fixed in this
    example. We consider mean/std parameters to be the parameters
    of a log-normal distribution in order to avoid +/- parameter 
    ambiguities that yield the same shape.

    Params
    ------
    n: int
        Number of supershape parameter samples to generate
    mean: array
        Log-normal frequency mean for m1/m2 parameter
    std: array
        Log-normal standard deviation for m1/m2 parameter
    '''
    samples = torch.tensor([
        [0, 1, 1, 3, 3, 3],
        [0, 1, 1, 3, 3, 3],
    ]).float().view(1,2,6).repeat(n,1,1)
    m1 = torch.empty(n).log_normal_(mean[0], std[0])
    m2 = torch.empty(n).log_normal_(mean[1], std[1])
    samples[:, 0, 0] = m1
    samples[:, 1, 0] = m2
    return samples

def update_simulations(remote_sims, n, mean, std):
    '''Updates all remote simulations with new shape parameter samples.
    
    We split N samples into N//R chunks where R is the number of
    simulation instances.
    '''
    samples = generate_samples(n, mean, std)
    R = len(remote_sims)
    for remote, subset in zip(remote_sims, torch.chunk(samples, R)):
        remote.send(shape_params=subset.numpy())

def item_transform(item):
    '''Transformation applied to each received simulation item.

    Here we exctract the image, normalize it and return it together
    with useful meta-data.
    '''
    x = item['image'].astype(np.float32)
    x = (x - 127.5) / 127.5
    return np.transpose(x, (2, 0, 1)),  item['params'], item['btid']

def get_target_images(dl, remotes, n=128):
    '''Returns a set of images from the target distribution.'''
    update_simulations(
        remotes, 
        n, 
        torch.tensor([MEAN_TARGET, MEAN_TARGET]), 
        torch.tensor([STD_TARGET, STD_TARGET]))
    images = []
    gen = iter(dl)
    for _ in range(n//BATCH):
        (img, params, btid) = next(gen)
        images.append(img)     
    return data.TensorDataset(torch.tensor(np.concatenate(images, 0)))

def infinite_batch_generator(dl):
    '''Generate infinite number of batches from a dataloader.'''
    while True:
        for data in dl:
            yield data

class Discriminator(nn.Module):
    '''Image descriminator.

    The task of the discriminator is to distinguish images from the target
    distribution from those of the simulator distribution. In the beginning
    this is easy, as the target distribution is quite narrow, while the
    simulator is producing images of supershapes from large spectrum. During
    optimization of the simulation parameters the classification of images
    will get continously harder as the simulation parameters are tuned
    towards the (unkown) target distribution parameters.
    
    The discriminator weights are trained via backpropagation.
    '''

    def __init__(self):
        super().__init__()
        ndf = 32
        nc = 3
        self.features = nn.Sequential(
            # state size. (ndf) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(self._weights_init)

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        return x.view(-1, 1).squeeze(1)


def log_probs(theta_mean, theta_std, samples):
    '''Returns the log-probabilities of the given samples w.r.t the log-normal distributions.'''
    ln0 = LogNormal(theta_mean[0], torch.exp(theta_std[0]))
    ln1 = LogNormal(theta_mean[1], torch.exp(theta_std[1]))
    return (
        ln0.log_prob(samples[:, 0]),
        ln1.log_prob(samples[:, 1]),
    )



def main():

    # Define how we want to launch Blender
    launch_args = dict(
        scene=Path(__file__).parent/'supershape.blend',
        script=Path(__file__).parent/'supershape.blend.py',
        num_instances=SIM_INSTANCES, 
        named_sockets=['DATA', 'CTRL'],
    )

    # Create an untrained discriminator.
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netD = Discriminator().to(dev)

    # Launch Blender
    with btt.BlenderLauncher(**launch_args) as bl:
        # Create remote dataset
        addr = bl.launch_info.addresses['DATA']
        sim_ds = btt.RemoteIterableDataset(addr, item_transform=item_transform)        
        sim_dl = data.DataLoader(sim_ds, batch_size=BATCH, num_workers=0, shuffle=False)

        # Create a control channel to each Blender instance. We use this channel to 
        # communicate new shape parameters to be rendered.
        addr = bl.launch_info.addresses['CTRL']
        remotes = [btt.DuplexChannel(a) for a in addr]

        # Fetch images of the target distribution. In the following we assume the 
        # target distribution to be unknown.
        target_ds = get_target_images(sim_dl, remotes, n=BATCH)
        target_dl = data.DataLoader(target_ds, batch_size=BATCH, num_workers=0, shuffle=True)
       
        # Initial simulation parameters. The parameters in mean and std are off from the target
        # distribution parameters. Note that we especially enlarge the scale of the distribution
        # to get explorative behaviour in the beginning.
        theta_mean = torch.tensor([1.2, 3.0], requires_grad=True)
        theta_std = torch.log(torch.tensor([STD_TARGET*4, STD_TARGET*4])).requires_grad_() # initial scale has to be larger the farther away we assume to be from solution.

        # Setup discriminator and simulation optimizer
        optD = optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.999))
        optS = optim.Adam([theta_mean, theta_std], lr=5e-2, betas=(0.7, 0.999))

        # Get generators for image batches from target and simulation.
        gen_real = infinite_batch_generator(target_dl)
        gen_sim = infinite_batch_generator(sim_dl)
        crit = nn.BCELoss(reduction='none')
        
        epoch = 0
        b = 0.          # baseline to reduce variance of gradient estimator.        
        balpha = 0.95   # baseline exponential smoothing factor.
        first = True

        # Send instructions to render supershapes from the starting point.
        update_simulations(remotes, BATCH, theta_mean.detach(), torch.exp(theta_std.detach()))
        for (real, sim) in zip(gen_real, gen_sim):
            ### Train the discriminator from target and simulation images.
            label = torch.full((BATCH,), TARGET_LABEL, dtype=torch.float32, device=dev)
            netD.zero_grad()
            target_img = real[0].to(dev)
            output = netD(target_img)
            errD_real = crit(output, label)
            errD_real.mean().backward()
            D_real = output.mean().item()

            sim_img, sim_param, sim_btid = sim
            sim_img = sim_img.to(dev)
            label.fill_(SIM_LABEL)
            output = netD(sim_img)
            errD_sim = crit(output, label)
            errD_sim.mean().backward()
            D_sim = output.mean().item()
            if (D_real - D_sim) < 0.95:
                optD.step()
                print('D step: mean real', D_real, 'mean sim', D_sim)

            ### Optimize the simulation parameters.
            # We update the simulation parameters once the discriminator
            # has started to converge. Note that unlike to GANs the generator 
            # (simulation) is giving meaningful output from the very beginning, so we
            # give the discriminator some time to adjust and avoid spurious signals
            # in gradient estimation of the simulation parameters.
            #
            # Note, the rendering function is considered a black-box and we cannot
            # propagate through it. Therefore we reformulate the optimization as
            # minimization of an expectation with the parameters in the distribution
            # the expectation runs over. Using score-function gradients permits gradient
            # based optimization _without_ access to gradients of the render function.
            if not first or (D_real - D_sim) > 0.7:
                optS.zero_grad()
                label.fill_(TARGET_LABEL)
                with torch.no_grad():
                    output = netD(sim_img)
                    errS_sim = crit(output, label)
                    GD_sim = output.mean().item()

                lp = log_probs(theta_mean, theta_std, sim_param[...,0].view(-1,2))
                loss = lp[0] * (errS_sim.cpu() - b) + lp[1] * (errS_sim.cpu() - b)
                loss.mean().backward()
                optS.step()

                if first:
                    b = errS_sim.mean().detach()
                else:
                    b = balpha * errS_sim.mean().detach() + (1-balpha)*b
                   
                print('S step:', theta_mean.detach().numpy(), torch.exp(theta_std).tolist(), 'mean sim', GD_sim)
                first = False        

            # Generate shapes according to updated parameters.
            update_simulations(remotes, BATCH, theta_mean.detach(), torch.exp(theta_std.detach()))
                
            epoch += 1
            if epoch % 10 == 0:
                vutils.save_image(target_img, 'tmp/real.png', normalize=True)
                vutils.save_image(sim_img, 'tmp/sim_samples_%03d.png' % (epoch), normalize=True)
                vutils.save_image(sim_img[sim_btid==0], 'tmp/sim_samples_mask_%03d.png' % (epoch), normalize=True)

if __name__ == '__main__':
    main()
