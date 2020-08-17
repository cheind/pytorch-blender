## Guided Domain Randomization

This example demonstrates optimizing simulation parameters towards target distribution parameters by minimizing the expected loss between target and simulation images measured by a learnable discriminator function. We consider the target distribution parameters to be unknown, but assume access to a set of target images.

In this example we render images of 3D parametric [supershapes](https://en.wikipedia.org/wiki/Superformula). A supershape can be described by 12 parameters and its appearance varies greatly along with the parameters (spheres, cubes, flower-like,...). In particular, the optimization tries to adjust the parameters 'm1' and 'm2' in such a way that the discriminator is not able to distinguish whether the given image is more likely to come from the simulation distribution or the target distribution. Once the discriminator reaches this state, the optimization parameters have usually converged towards the true target parameters.

Note, we can frame this example in a GAN pattern: the generator consists of the simulation + probabilistic model governing the shape. The discriminator is a neural network attempting to distinguish between images of the target distribution and those to the simulation distribution. However, there is at least one **key difference** to GANs: the render function (Blender) is a black-box without access to gradients. We therefore frame the optimization as a minimization over an expected discriminator loss. 

The image series below shows target images (left) and optimization progress at step (10,40,80).
<p align="center">
<img src="etc/real.png" width="200">
<img src="etc/sim_samples_010.png" width="200">
<img src="etc/sim_samples_040.png" width="200">
<img src="etc/sim_samples_080.png" width="200">
</p>

### Run

To recreate these results run [guideddr.py](./guideddr.py) as follows
```
python guideddr.py
```
which will generate output images in `./tmp/` and output the optimization progress on the console 
```
D step: mean real 0.7307174801826477 mean sim 0.6040560007095337
D step: mean real 0.6737353205680847 mean sim 0.29735732078552246
D step: mean real 0.825548529624939 mean sim 0.2806171178817749
D step: mean real 0.8810040950775146 mean sim 0.1778283715248108
S step: [1.25 2.95] [0.4205084443092346, 0.4205084443092346] mean sim 0.13082975149154663
D step: mean real 0.9086431860923767 mean sim 0.21175092458724976
S step: [1.2831218 2.9007795] [0.4404144883155823, 0.4418575167655945] mean sim 0.14274010062217712
...
```
The true parameters being around 2.5/2.5 with a standard deviation of 0.1/0.1.

### Dependencies

Besides **blendtorch**, this examples requires [this tiny supershape](https://github.com/cheind/supershape) library to be acessible from within Blender (update `PYTHONPATH` prior executing the example).



