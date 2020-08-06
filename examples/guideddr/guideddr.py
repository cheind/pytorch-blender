from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.distributions import MultivariateNormal, Normal
from blendtorch import btt
import torchvision.utils as vutils

def item_transform(item):
    x = item['image'].astype(np.float32)
    x = (x - 127.5) / 127.5
    return np.transpose(x, (2, 0, 1)),  item['params'], item['mid']

def get_real_images(ds, duplex, m1true, m2true, n=128):
    mid = duplex.mid
    duplex.send(dict(m1=m1true, m2=m2true, mid=mid))
    images = []
    mids = []
    for (img, params, rmid) in ds:
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
        ndf = 64
        nc = 3
        self.features = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
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

BATCH = 32
REAL_LABEL = 1
SIM_LABEL = 0

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

        real_ds = get_real_images(sim_ds, duplex, m1true=10, m2true=10, n=BATCH*4)
        real_dl = data.DataLoader(real_ds, batch_size=BATCH, num_workers=0, shuffle=True)

        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        netD = Discriminator().to(dev)
        netD.apply(weights_init)

        last_mid = duplex.mid
        sim_params = torch.tensor([5.0, 5.0], requires_grad=True)
        duplex.send(dict(m1=sim_params[0].item(), m2=sim_params[1].item(), mid=last_mid))

        optD = optim.Adam(netD.parameters(), lr=1e-6, betas=(0.5, 0.999))
        optS = optim.SGD([sim_params], lr=1e-2)

        gen_real = infinite_batch_generator(real_dl)
        gen_sim = infinite_batch_generator(sim_dl)
        crit = nn.BCELoss(reduction='none')
        
        e = 0
        while True:
            sim_img, sim_param, sim_mid = next(gen_sim)
            mask = (sim_mid == last_mid)
            have_sim = mask.sum() > BATCH//2

            if have_sim:                
                label = torch.full((BATCH,), REAL_LABEL, dtype=torch.float32, device=dev)

                # Update D with real
                netD.zero_grad()
                real_img = next(gen_real)[0].to(dev)
                output = netD(real_img)
                errD_real = crit(output, label)
                errD_real.mean().backward()
                D_real = output.mean().item()

                # Update D with sim
                label.fill_(SIM_LABEL)
                output = netD(sim_img.to(dev)[mask])
                errD_sim = crit(output, label[mask])
                errD_sim.mean().backward()
                D_sim = output.mean().item()
                optD.step()

                # Update sim params
                optS.zero_grad()
                label.fill_(REAL_LABEL)
                errS_sim = crit(output, label[mask])
                E_sim = errS_sim.mean().item()
                mv0 = Normal(sim_params[0], 0.02)
                sf0 = mv0.log_prob(sim_param[mask][:, 0]) * errS_sim.detach().cpu()
                sf0.mean().backward()

                mv1 = Normal(sim_params[1], 0.02)
                sf1 = mv1.log_prob(sim_param[mask][:, 1]) * errS_sim.detach().cpu()
                sf1.mean().backward()

                sim_params.grad.mul_(E_sim * 10)

                optS.step()

                print('sim params', sim_params)

                last_mid = duplex.mid
                duplex.send(dict(m1=sim_params[0].item(), m2=sim_params[1].item(), mid=last_mid))

            e += 1
            if e % 20 == 0:
                vutils.save_image(real_img, 'tmp/real.png', normalize=True)
                vutils.save_image(sim_img, 'tmp/sim_samples_%03d.png' % (e), normalize=True)

            print('mean D real', D_real, 'mean D sim', D_sim, 'err_sim', E_sim)

if __name__ == '__main__':
    main()
