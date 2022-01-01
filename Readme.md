# blendtorch
[![Build Status](https://app.travis-ci.com/cheind/pytorch-blender.svg?branch=develop)](https://app.travis-ci.com/cheind/pytorch-blender)

**blendtorch** is a Python framework to seamlessly integrate Blender into PyTorch for deep learning from artificial visual data. We utilize Eevee, a new physically based real-time renderer, to synthesize images and annotations in real-time and thus avoid stalling model training in many cases. If you find the project helpful, you consider [citing](#cite_anchor) it.

Feature summary
 - ***Data Generation***: Stream distributed Blender renderings directly into PyTorch data pipelines in real-time for supervised learning and domain randomization applications. Supports arbitrary pickle-able objects to be send alongside images/videos. Built-in recording capability to replay data without Blender. Bi-directional communication channels allow Blender simulations to adapt during network training. </br>More info [\[examples/datagen\]](examples/datagen), [\[examples/compositor_normals_depth\]](examples/compositor_normals_depth),  [\[examples/densityopt\]](examples/densityopt)
 - ***OpenAI Gym Support***: Create and run remotely controlled Blender gyms to train reinforcement agents. Blender serves as simulation, visualization, and interactive live manipulation environment.
 </br>More info [\[examples/control\]](examples/control)

The figure below visualizes the basic concept of **blendtorch** used in the context of generating artificial training data for a real-world detection task.

<div align="center">
<img src="etc/blendtorch_intro_v3.svg" width="90%">
</div>

## Getting started
 1. Read the installation instructions below
 1. To get started with **blendtorch** for training data training read [\[examples/datagen\]](examples/datagen). 
 1. To learn about using **blendtorch** for creating reinforcement training environments read [\[examples/control\]](examples/control).

## Installation

**blendtorch** is composed of two distinct sub-packages: `bendtorch.btt` (in [pkg_pytorch](./pkg_pytorch)) and `blendtorch.btb` (in [pkg_blender](./pkg_blender)), providing the PyTorch and Blender views on **blendtorch**. 

### Prerequisites
This package has been tested with
 - [Blender](https://www.blender.org/) >= 2.83/2.91/3.0 (Python 3.7/3.9)
 - [PyTorch](http://pytorch.org) >= 1.5/1.10 (Python 3.7/3.8)
running Windows 10 and Linux.

Other versions might work as well, but have not been tested. 

### Clone this repository
```
git clone https://github.com/cheind/pytorch-blender.git <DST>
```

### Extend `PATH`
Ensure Blender executable is in your environments lookup `PATH`. On Windows this can be accomplished by
```
set PATH=c:\Program Files\Blender Foundation\Blender 2.91;%PATH%
```

On Ubuntu when blender is [installed using snap](https://snapcraft.io/install/blender/ubuntu), the path may be included by adding the following line to your ~/.bashrc,

```
export PATH=/snap/blender/current/${PATH:+:${PATH}}
```

### Complete Blender settings
Open Blender at least once, and complete the initial settings. If this step is missed, some of the tests (especially the tests relating RL) will fail (Blender 2.91).

### Install **blendtorch** Blender part
```
blender --background --python <DST>/scripts/install_btb.py
```
installs `blendtorch-btb` into the Python environment bundled with Blender. 

### Install **blendtorch** PyTorch part
```
pip install -e <DST>/pkg_pytorch
```
installs `blendtorch-btt` into the Python environment that you intend to run PyTorch from. While not required, it is advised to install OpenAI gym if you intend to use **blendtorch** for reinforcement learning
```
pip install gym
```

### Developer instructions
This step is optional. If you plan to run the unit tests
```
pip install -r requirements_dev.txt
pytest tests/
```

### Troubleshooting
Run
```
blender --version
```
and check if the correct Blender version (>=2.83) is written to console. Next, ensure that `blendtorch-btb` installed correctly
```
blender --background --python-use-system-env --python-expr "import blendtorch.btb as btb; print(btb.__version__)"
```
which should print **blendtorch** version number on success. Next, ensure that `blendtorch-btt` installed correctly
```
python -c "import blendtorch.btt as btt; print(btt.__version__)"
```
which should print **blendtorch** version number on success.

## Architecture
Please see [\[examples/datagen\]](examples/datagen) and [\[examples/control\]](examples/control) for an in-depth architectural discussion. Bi-directional communication is explained in [\[examples/densityopt\]](examples/densityopt).

## Runtimes

The following tables show the mean runtimes per batch (8) and per image for a simple Cube scene (640x480xRGBA). See [benchmarks/benchmark.py](./benchmarks/benchmark.py) for details. The timings include rendering, transfer, decoding and batch collating. Reported timings are for Blender 2.8. Blender 2.9 performs equally well on this scene, but is usually faster for more complex renderings.

| Blender Instances  | Runtime sec/batch | Runtime sec/image | Arguments|
|:-:|:-:|:-:|:-:|
| 1  | 0.236 | 0.030| UI refresh|
| 2  | 0.14 | 0.018| UI refresh|
| 4  | 0.099 | 0.012| UI refresh|
| 5  | 0.085 | 0.011| no UI refresh|

Note: If no image transfer is needed, i.e in reinforcement learning of physical simulations, 2000Hz are easily achieved.

<a name="cite_anchor"></a>
## Cite
The code accompanies our academic work [[1]](https://arxiv.org/abs/1907.01879),[[2]](https://arxiv.org/abs/2010.11696) in the field of machine learning from artificial images. Please consider the following publications when citing **blendtorch**
```
@inproceedings{blendtorch_icpr2020_cheind,
    author = {Christoph Heindl, Lukas Brunner, Sebastian Zambal and Josef Scharinger},
    title = {BlendTorch: A Real-Time, Adaptive Domain Randomization Library},
    booktitle = {
        1st Workshop on Industrial Machine Learning 
        at International Conference on Pattern Recognition (ICPR2020)
    },
    year = {2020},
}

@inproceedings{robotpose_etfa2019_cheind,
    author={Christoph Heindl, Sebastian Zambal, Josef Scharinger},
    title={Learning to Predict Robot Keypoints Using Artificially Generated Images},
    booktitle={
        24th IEEE International Conference on 
        Emerging Technologies and Factory Automation (ETFA)
    },    
    year={2019}
}
```

## Caveats
- Despite offscreen rendering is supported in Blender 2.8x it requires a UI frontend and thus cannot run in `--background` mode. If your application does not require offscreen renderings you may enable background usage (see [tests/](tests/) for examples).
- The renderings produced by Blender are by default in linear color space and thus will appear darker than expected when displayed.
