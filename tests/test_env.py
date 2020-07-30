import pytest
from pathlib import Path
from blendtorch import btt

BLENDDIR = Path(__file__).parent/'blender'

class MyEnv(btt.env.OpenAIRemoteEnv):
    def __init__(self, background=True, **kwargs):
        super().__init__(version='1.0.0')
        self.launch(scene='', script=BLENDDIR/'env.blend.py', background=background, **kwargs)

def _run_remote_env(background):
    env = MyEnv(background=background)

    obs = env.reset()
    assert obs==0.
    obs, reward, done, info = env.step(0.1)
    assert obs== pytest.approx(0.1)
    assert reward==0.
    assert not done
    assert info['count']==2 # 1 is already set by reset()
    obs, reward, done, info = env.step(0.6)
    assert obs== pytest.approx(0.6)
    assert reward==1.
    assert not done
    assert info['count']==3
    for _ in range(8):
        obs, reward, done, info = env.step(0.6)
    assert done
    
    obs = env.reset()
    assert obs==0.
    obs, reward, done, info = env.step(0.1)
    assert obs== pytest.approx(0.1)
    assert reward==0.
    assert not done
    assert info['count']==2

    env.close()        

@pytest.mark.background
def test_remote_env():
    _run_remote_env(background=True)
    _run_remote_env(background=False)
    
    