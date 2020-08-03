# blendtorch

**blendtorch** is a Python framework to seamlessly integrate [Blender](http://blender.orf) renderings into [PyTorch](http://pytorch.org) datasets for deep learning from artificial visual data. We utilize Eevee, a new physically based real-time renderer, to synthesize images and annotations in real-time and thus avoid stalling model training in many cases.

Feature summary
 - ***Data Streaming***: Stream distributed Blender renderings directly into PyTorch data pipelines in real-time for supervised learning and domain randomization applications. Supports arbitrary pickle-able objects to be send alongside images/videos. Built-in recording capability to replay data without Blender.</br>More info [\[examples/datagen\]](examples/datagen)
 - ***OpenAI Gym Support***: Create and run remotely controlled Blender gyms to train reinforcement agents. Blender serves as simulation, visualization, and interactive live manipulation environment.
 </br>More info [\[examples/control\]](examples/control)

The figure below visualizes a single image/label batch received by PyTorch from four parallel Blender instances. Each Blender process repeatedly performs motion simulations of randomized cubes.

<p align="center">
<img src="etc/result_physics.png" width="500">
</p>

## Getting started
 1. Read the installation instructions below
 1. To get started with **blendtorch** for training data training read [\[examples/datagen\]](examples/datagen). 
 1. To learn about using **blendtorch** for creating reinforcement training environments read [\[examples/control\]](examples/control).

## Installation

### Prerequisites
This package has been tested using the following packages
 - [Blender](https://www.blender.org/) >= 2.83 (Python 3.7)
 - [PyTorch](http://pytorch.org) >= 1.50 (Python 3.7/3.8)

Other versions might work as well, but have not been tested. 

### Clone this repository
```
git clone https://github.com/cheind/pytorch-blender.git <DST>
```

### Extend `PATH`
Next, ensure Blender executable is in your environments lookup `PATH`. On Windows this can be accomplished by
```
set PATH=c:\Program Files\Blender Foundation\Blender 2.83;%PATH%
```

### Install `blendtorch-btb` 
```
blender --background --python <DST>/scripts/install_btb.py
```
installs `blendtorch-btb` into the Python environment bundled with Blender. 

### Install `blendtorch-btt` 
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
Please see [\[examples/datagen\]](examples/datagen) and [examples/control\]](examples/control) for an in-depth discussion.

## Cite
The code accompanies our [academic work](https://arxiv.org/abs/1907.01879) in the field of machine learning from artificial images. When using please cite the following work
```
@inproceedings{robotpose_etfa2019_cheind,
    author={Christoph Heindl and Sebastian Zambal and Josef Scharinger},
    title={Learning to Predict Robot Keypoints Using Artificially Generated Images},
    booktitle={
        24th IEEE International Conference on 
        Emerging Technologies and Factory Automation (ETFA)
    },    
    year={2019},
    pages={1536-1539},
    doi={10.1109/ETFA.2019.8868243},
    isbn={978-1-7281-0303-7},
}
```

## Runtimes
The following tables show the mean runtimes per batch (8) and per image for a simple Cube scene (640x480xRGBA). See [benchmarks/benchmark.py](./benchmarks/benchmark.py) for details. The timings include rendering, transfer, decoding and batch collating.

| Blender Instances  | Runtime sec/batch | Runtime sec/image | Arguments|
|:-:|:-:|:-:|:-:|
| 1  | 0.236 | 0.030| UI refresh|
| 2  | 0.14 | 0.018| UI refresh|
| 4  | 0.099 | 0.012| UI refresh|
| 5  | 0.085 | 0.011| no UI refresh|

Note: If no image transfer is needed, i.e in reinforcement learning of physical simulations, 2000Hz are easily achieved.

## Caveats
- Despite offscreen rendering is supported in Blender 2.8x it requires a UI frontend and thus cannot run in `--background` mode.
- The renderings produced by Blender are by default in linear color space and thus will appear darker than expected when displayed.
