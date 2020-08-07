from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.distributions import MultivariateNormal, Normal, LogNormal
from blendtorch import btt
import torchvision.utils as vutils

BATCH = 64
REAL_LABEL = 1
SIM_LABEL = 0

LOG_MEAN_TRUE = 2.25 # rough sample range: 8.5-10.5
LOG_STD_TRUE = 0.1

def item_transform(item):
    x = item['image'].astype(np.float32)
    x = (x - 127.5) / 127.5
    return np.transpose(x, (2, 0, 1)),  item['params'], item['mid'], item['btid']

def get_real_images(ds, duplex, n=128):
    mid = duplex.send(type='lognormal', mean=[LOG_MEAN_TRUE, LOG_MEAN_TRUE], std=[LOG_STD_TRUE, LOG_STD_TRUE])
    images = []
    mids = []
    for (img, params, rmid, btid) in ds:
        if rmid == mid:
            images.append(img)
            mids.append(mid)
        if len(images) == n:
            break

    return data.TensorDataset(torch.tensor(images))

def infinite_batch_generator(dl):
    while True:
        for data in dl:
            yield data

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 32
        nc = 3
        self.features = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)




def main():

    # Define how we want to launch Blender
    launch_args = dict(
        scene=Path(__file__).parent/'supershape.blend',
        script=Path(__file__).parent/'supershape.blend.py',
        num_instances=1, 
        named_sockets=['DATA', 'CTRL'],
    )

    # Launch Blender
    with btt.BlenderLauncher(**launch_args) as bl:
        # Create remote dataset and limit max length to 16 elements.
        addr = bl.launch_info.addresses['DATA']
        sim_ds = btt.RemoteIterableDataset(addr, max_items=100000, item_transform=item_transform)        
        sim_dl = data.DataLoader(sim_ds, batch_size=BATCH, num_workers=2, shuffle=False)

        addr = bl.launch_info.addresses['CTRL']
        duplex = btt.DuplexChannel(addr[0])
        #duplex_uniform = btt.DuplexChannel(addr[1])
        #duplex_uniform.send(type='uniform', low=0.0, high=20.0)

        real_ds = get_real_images(sim_ds, duplex, n=BATCH*2)
        real_dl = data.DataLoader(real_ds, batch_size=BATCH, num_workers=0, shuffle=True)

        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        netD = Discriminator().to(dev)
        netD.apply(weights_init)

        sim_params = torch.tensor([1.5, 1.5], requires_grad=True)
        last_mid = duplex.send(type='lognormal', mean=sim_params.tolist(), std=[LOG_STD_TRUE*4, LOG_STD_TRUE*4])

        optD = optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5, 0.999))
        optS = optim.Adam([sim_params], lr=1e-1)

        gen_real = infinite_batch_generator(real_dl)
        gen_sim = infinite_batch_generator(sim_dl)
        crit = nn.BCELoss(reduction='none')
        
        epoch = 0
        b = 0.
        first = True
        baseline = True
        alpha = 0.98
        while True:
            
            label = torch.full((BATCH,), REAL_LABEL, dtype=torch.float32, device=dev)
            # Update D with real
            netD.zero_grad()
            real_img = next(gen_real)[0].to(dev)
            output = netD(real_img)
            errD_real = crit(output, label)
            errD_real.mean().backward()
            D_real = output.mean().item()

            # Update D with sim
            sim_img, sim_param, sim_mid, sim_btid = next(gen_sim)
            label.fill_(SIM_LABEL)
            output = netD(sim_img.to(dev))
            errD_sim = crit(output, label)
            errD_sim.mean().backward()
            D_sim = output.mean().item()
            if epoch % 5 == 0:
                optD.step()
            print('mean D real', D_real, 'mean D sim', D_sim)

            mask = (sim_mid == last_mid)
            have_sim = mask.sum() > BATCH//4
            if have_sim and epoch > 20:
                # Update sim params, called only every 20epochs or so.
                # blender generates new data meanwhile with old MID.
                optS.zero_grad()
                label.fill_(REAL_LABEL)
                with torch.no_grad():
                    output = netD(sim_img.to(dev)[mask])
                    errS_sim = crit(output, label[mask])
                    GD_sim = output.mean().item()

                ln0 = LogNormal(sim_params[0], LOG_STD_TRUE*4)
                sf = ln0.log_prob(sim_param[mask][:, 0]) * (errS_sim.cpu() - b)
                sf.mean().backward(retain_graph=True)

                ln1 = LogNormal(sim_params[1], LOG_STD_TRUE*4)
                sf = ln1.log_prob(sim_param[mask][:, 1]) * (errS_sim.cpu() - b)
                sf.mean().backward()

                if baseline:
                    if first:
                        b = errS_sim.mean().detach()
                    else:
                        b = alpha * errS_sim.mean().detach() + (1-alpha)*b
                optS.step()                  

                print('sim params', sim_params, 'mean D sim', GD_sim)
                last_mid = duplex.send(type='lognormal', mean=sim_params.tolist(), std=[LOG_STD_TRUE*4, LOG_STD_TRUE*4])        
                first = False        
                
            epoch += 1
            if epoch % 10 == 0:
                vutils.save_image(real_img, 'tmp/real.png', normalize=True)
                vutils.save_image(sim_img, 'tmp/sim_samples_%03d.png' % (epoch), normalize=True)
                vutils.save_image(sim_img[sim_btid==0], 'tmp/sim_samples_mask_%03d.png' % (epoch), normalize=True)
                

            

if __name__ == '__main__':
    main()
