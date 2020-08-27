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
DEFAULT_MEAN_TARGET = 2.25 # rough sample range: 8.5-10.5
'''Long./Lat. log-normal supershape frequency (m1,m2) target standard deviation'''
DEFAULT_STD_TARGET = 0.1 
'''Baseline (control variate) smoothing factor'''
BASELINE_ALPHA = 0.9

class ProbModel(nn.Module):
    '''Probabilistic model governing supershape parameters.

    In this example, we model the shape m1/m2 as random variables. We assume
    independence and associate a log-normal distribution for each of them. We choose
    in order to avoid +/- parameter ambiguities that yield the same shape.

        p(m1,m2) = p(m1)p(m2) with 
        p(m1) = LogNormal(mu_m1, std_m1),
        p(m2) = LogNormal(mu_m2, std_m2)

    We consider the mean/scale of each distribution to be parameters subject to
    optimization. Note, we model the scale parameter as log-scale to allow 
    unconstrained (scale > 0) optimization.
    '''

    def __init__(self, m1m2_mean, m1m2_std):
        super().__init__()
        
        self.m1m2_mean = nn.Parameter(torch.as_tensor(m1m2_mean).float(), requires_grad=True)
        self.m1m2_log_std = nn.Parameter(torch.log(torch.as_tensor(m1m2_std).float()), requires_grad=True)

    def sample(self, n):
        '''Returns n samples.'''
        m1,m2 = self.dists
        return {
            'm1': m1.sample((n,)),
            'm2': m2.sample((n,)),
        }
    
    def log_prob(self, samples):
        '''Returns the joint log-probabilities of the given samples.'''
        m1,m2 = self.dists
        return m1.log_prob(samples['m1']) + m2.log_prob(samples['m2'])

    @property
    def dists(self):
        '''Returns the parametrized distributions for m1/m2.'''
        # Creating the distributions always on the fly, otherwise we get
        # PyTorch warnings about differentiating a second time.
        return (
            LogNormal(self.m1m2_mean[0], torch.exp(self.m1m2_log_std[0])),
            LogNormal(self.m1m2_mean[1], torch.exp(self.m1m2_log_std[1]))
        )  

    def readable_params(self):
        '''Helper function to access params as vector.'''
        return torch.cat([self.m1m2_mean.detach(), torch.exp(self.m1m2_log_std).detach()])

    @staticmethod
    def to_supershape(samples):
        '''Converts m1/m2 samples to full supershape parameters.
        
        We assume all parameter except for m1/m2 to be fixed in this
        example.
        '''
        N = samples['m1'].shape[0]
        params = samples['m1'].new_tensor([
            [0, 1, 1, 3, 3, 3],
            [0, 1, 1, 3, 3, 3],
        ]).float().view(1,2,6).repeat(N,1,1)
        params[:, 0, 0] = samples['m1'].detach()
        params[:, 1, 0] = samples['m2'].detach()
        return params

def update_simulations(remote_sims, params):
    '''Updates all remote simulations with new supershape samples.
    
    We split N parameter samples into N/R chunks where R is the number of
    simulation instances. Besides the parameters, we send subset indices
    to the simulation instances which will be returned to us alongside 
    with the rendered images. The subset indices allow us to associate
    parameters with images in the optimization.
    '''
    ids = torch.arange(params.shape[0]).long()
    R = len(remote_sims)
    for remote, subset, subset_ids in zip(remote_sims, torch.chunk(params, R), torch.chunk(ids, R)):
        remote.send(shape_params=subset.cpu().numpy(), shape_ids=subset_ids.numpy())

def item_transform(item):
    '''Transformation applied to each received simulation item.

    Extract the image, normalize it and return it together
    with meta-data. In particular we return the `shape_id` which
    allows us to associate the received item with a parameter 
    sample generated earlier.
    '''
    x = item['image'].astype(np.float32)
    x = (x - 127.5) / 127.5
    return np.transpose(x, (2, 0, 1)), item['shape_id']

def get_target_images(dl, remotes, mu_m1m2, std_m1m2, n):
    '''Returns a set of images from the target distribution.'''
    pm = ProbModel(mu_m1m2, std_m1m2)
    samples = pm.sample(n)
    update_simulations(remotes, ProbModel.to_supershape(samples))
    images = []
    gen = iter(dl)
    for _ in range(n//BATCH):
        (img, shape_id) = next(gen)
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

def run(args):

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
        if args.random_target:
            mu_m1m2_target = np.random.uniform(0.1, 20, size=2).astype(np.float32)
            std_m1m2_target = np.random.uniform(0.1, 0.3, size=2).astype(np.float32)
        else:
            mu_m1m2_target = [DEFAULT_MEAN_TARGET, DEFAULT_MEAN_TARGET]
            std_m1m2_target = [DEFAULT_STD_TARGET, DEFAULT_STD_TARGET]
        print('Target params:', mu_m1m2_target, std_m1m2_target)  

        target_ds = get_target_images(sim_dl, remotes, mu_m1m2_target, std_m1m2_target, n=BATCH)
        target_dl = data.DataLoader(target_ds, batch_size=BATCH, num_workers=0, shuffle=True)
       
        # Initial simulation parameters. The parameters in mean and std are off from the target
        # distribution parameters. Note that we especially enlarge the scale of the initial 
        # distribution to get explorative behaviour in the beginning.
        if args.random_start:
            mu_m1m2 = np.asarray(mu_m1m2_target) + np.random.randn(2)
            std_m1m2 = [std_m1m2_target[0]*4, std_m1m2_target[0]*4]            
        else:
            mu_m1m2 = [1.2, 3.0]
            std_m1m2 = [DEFAULT_STD_TARGET*4, DEFAULT_STD_TARGET*4]
        pm = ProbModel(mu_m1m2, std_m1m2)

        # Setup discriminator and simulation optimizer
        optD = optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.999))
        optS = optim.Adam(pm.parameters(), lr=5e-2, betas=(0.7, 0.999))

        # Get generators for image batches from target and simulation.
        gen_real = infinite_batch_generator(target_dl)
        gen_sim = infinite_batch_generator(sim_dl)
        crit = nn.BCELoss(reduction='none')
        
        epoch = 0
        b = 0.          # Baseline to reduce variance of gradient estimator.
        first = True
        param_history = []

        # Send instructions to render supershapes from the starting point.
        samples = pm.sample(BATCH)
        update_simulations(remotes, pm.to_supershape(samples))
        for (real, sim) in zip(gen_real, gen_sim):
            ### Train the discriminator from target and simulation images.
            label = torch.full((BATCH,), TARGET_LABEL, dtype=torch.float32, device=dev)
            netD.zero_grad()
            target_img = real[0].to(dev)
            output = netD(target_img)
            errD_real = crit(output, label)
            errD_real.mean().backward()
            D_real = output.mean().item()

            sim_img, sim_shape_id = sim
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

                log_probs = pm.log_prob(samples)
                loss = log_probs[sim_shape_id] * (errS_sim.cpu() - b)
                loss.mean().backward()
                optS.step()                

                if first:
                    b = errS_sim.mean()
                else:
                    b = BASELINE_ALPHA * errS_sim.mean() + (1-BASELINE_ALPHA)*b

                print('S step:', pm.m1m2_mean.detach().numpy(), torch.exp(pm.m1m2_log_std).detach().numpy(), 'mean sim', GD_sim)  
                first = False        
                del log_probs, loss

            # Generate shapes/images according to updated parameters.
            samples = pm.sample(BATCH)
            update_simulations(remotes, pm.to_supershape(samples))

            # Bookkeeping
            param_history.append(pm.readable_params())                
            epoch += 1
            if epoch % 5 == 0:
                vutils.save_image(target_img, 'tmp/real.png', normalize=True)
                vutils.save_image(sim_img, 'tmp/sim_samples_%03d.png' % (epoch), normalize=True)
            if epoch > args.num_epochs:
                # Append true target
                target = torch.tensor(
                    np.concatenate((mu_m1m2_target, std_m1m2_target))
                ).float()
                print('Abs.Diff to true params', abs(target - param_history[-1]))
                param_history.append(target)
                break

        param_history = torch.stack(param_history)
        return param_history
        

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--random-target', action='store_true')
    parser.add_argument('--random-start', action='store_true')
    parser.add_argument('--num-runs', default=1, type=int)
    parser.add_argument('--num-epochs', default=70, type=int)
    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d_%H%M%S")
    for i in range(args.num_runs):
        param_history = run(args)
        np.savetxt(
            f'tmp/run_{timestr}_{i:02d}_densityopt.txt', 
            param_history, 
            header='mu_m1, mu_m2, std_m1, std_m2', 
            comments='last entry corresponds to target params')