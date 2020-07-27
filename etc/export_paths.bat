@echo off
set PATH=c:\Program Files\Blender Foundation\Blender 2.83;%PATH%
set PYTHONPATH=%~dp0..\pkg_blender;%~dp0..\pkg_pytorch;%PYTHONPATH%
@echo on