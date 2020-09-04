import pytest
import multiprocessing as mp
from pathlib import Path
import json
from blendtorch import btt
from shlex import quote

BLENDDIR = Path(__file__).parent/'blender'
LAUNCH_ARGS = dict(
        scene='',
        script=str(BLENDDIR/'launcher.blend.py'), 
        num_instances=2,        
        named_sockets=['DATA', 'GYM'],
        background=True,
        instance_args=[['--x', '3'],['--x', '4']],
        seed=10
    )

def _validate_result(items):
    assert len(items) == 2
    first, second = 0, 1
    if items[0]['btid']==1:
        first, second = second, first
    
    assert items[first]['btargs']['btid']==0
    assert items[second]['btargs']['btid']==1
    assert items[first]['btargs']['btseed']==10
    assert items[second]['btargs']['btseed']==11
    assert items[first]['btargs']['btsockets']['DATA'].startswith('tcp://')
    assert items[first]['btargs']['btsockets']['GYM'].startswith('tcp://')
    assert items[second]['btargs']['btsockets']['DATA'].startswith('tcp://')
    assert items[second]['btargs']['btsockets']['GYM'].startswith('tcp://')
    assert items[first]['remainder'] == ['--x', '3']
    assert items[second]['remainder'] == ['--x', '4']


@pytest.mark.background
def test_launcher():    
    with btt.BlenderLauncher(**LAUNCH_ARGS) as bl:
        addr = bl.launch_info.addresses['DATA']
        ds = btt.RemoteIterableDataset(addr, max_items=2)
        items = [item for item in ds]
        _validate_result(items)


def _launch(q, tmp_path):
    with btt.BlenderLauncher(**LAUNCH_ARGS) as bl:
        path = Path(tmp_path / 'addresses.json')
        btt.LaunchInfo.save_json(path, bl.launch_info)
        q.put(path)
        bl.wait()

@pytest.mark.background
def test_launcher_connected_remote(tmp_path):
    # Simulates BlenderLauncher called from a separate process and
    # shows how one can connect to already launched instances through
    # serialization of addresses.
    q = mp.Queue()
    p = mp.Process(target=_launch, args=(q, tmp_path))
    p.start()
    path = q.get()
    launch_info = btt.LaunchInfo.load_json(path)
    ds = btt.RemoteIterableDataset(launch_info.addresses['DATA'], max_items=2)
    items = [item for item in ds]
    _validate_result(items)
    p.join()

def _launch_app(tmp_path):    
    from blendtorch.btt.apps import launch    
    with open(tmp_path/'launchargs.json', 'w') as fp:
        fp.write(json.dumps(LAUNCH_ARGS, indent=4))
    launch.main(['--out-launch-info', str(tmp_path/'launchinfo.json'), str(tmp_path/'launchargs.json')])

@pytest.mark.background
def test_launcher_app(tmp_path):

    p = mp.Process(target=_launch_app, args=(tmp_path,))
    p.start()

    import time
    path = tmp_path/'launchinfo.json'
    while not Path.exists(path):
        time.sleep(1)

    launch_info = btt.LaunchInfo.load_json(path)
    ds = btt.RemoteIterableDataset(launch_info.addresses['DATA'], max_items=2)
    items = [item for item in ds]
    _validate_result(items)
    
    p.join()
