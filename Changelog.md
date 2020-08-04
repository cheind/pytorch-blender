# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.0] - 2020-08-04
- Support for training RL agents in OpenAI environments defined in Blender. See `examples/control` for details.
- Support for dataset transformations when `num_workers > 0`.
- Support for message recording when `num_workers > 0`.
- Made `blendtorch.btb` and `blendtorch.btt` installable packages. See `Readme.md` for details.
- Remote data streaming now uses PyTorch IterableDataset and 
hence simplifies the interface. See `examples/datagen` for details.
- Added unit tests and CI.

## [0.1.0] - 2020-07-10
- added Blender Eevee 2.8 support