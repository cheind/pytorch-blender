from setuptools import setup
from pathlib import Path

THISDIR = Path(__file__).parent

with open(THISDIR / "requirements.txt") as f:
    required = f.read().splitlines()

with open(THISDIR / ".." / "Readme.md", encoding="utf-8") as f:
    long_description = f.read()

main_ns = {}
with open(THISDIR / "blendtorch" / "btt" / "version.py") as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="blendtorch-btt",
    author="Christoph Heindl and Sebastian Zambal",
    description="PyTorch part of project blendtorch. See also blendtorch-btb.",
    url="https://github.com/cheind/pytorch-blender",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=main_ns["__version__"],
    packages=["blendtorch.btt", "blendtorch.btt.apps"],
    install_requires=required,
    zip_safe=False,
    entry_points={
        "console_scripts": ["blendtorch-launch=blendtorch.btt.apps.launch:main"],
    },
)
