import subprocess
import os
import logging
import numpy as np
import psutil
import signal


from .finder import discover_blender
from .launch_info import LaunchInfo
from .utils import get_primary_ip

logger = logging.getLogger("blendtorch")


class BlenderLauncher:
    """Opens and closes Blender instances.

    This class is meant to be used withing a `with` block to ensure clean launch/shutdown of background processes.

    Params
    ------
    scene : str
        Scene file to be processed by Blender instances
    script: str
        Script file to be called by Blender
    num_instances: int (default=1)
        How many Blender instances to create
    named_sockets: list-like, optional
        Descriptive names of sockets to be passed to launched instanced
        via command-line arguments '-btsockets name=tcp://address:port ...'
        to Blender. They are also available via LaunchInfo to PyTorch.
    start_port : int (default=11000)
        Start of port range for publisher sockets
    bind_addr : str (default='127.0.0.1')
        Address to bind publisher sockets. If 'primaryip' binds to primary ip
        address the one with a default route or `127.0.0.1` if none is available.
    proto: string (default='tcp')
        Protocol to use.
    instance_args : array (default=None)
        Additional arguments per instance to be passed as command
        line arguments.
    blend_path: str, optional
        Additional paths to look for Blender
    seed: integer, optional
        Optional launch seed. Each instance will be given
    background: bool
        Launch Blender in background mode. Note that certain
        animation modes and rendering does require an UI and
        cannot run in background mode.

    Attributes
    ----------
    launch_info: LaunchInfo
        Process launch information, available after entering the
        context.
    """

    def __init__(
        self,
        scene,
        script,
        num_instances=1,
        named_sockets=None,
        start_port=11000,
        bind_addr="127.0.0.1",
        instance_args=None,
        proto="tcp",
        blend_path=None,
        seed=None,
        background=False,
    ):
        """Create BlenderLauncher"""
        self.num_instances = num_instances
        self.start_port = start_port
        self.bind_addr = bind_addr
        self.proto = proto
        self.scene = scene
        self.script = script
        self.blend_path = blend_path
        self.named_sockets = named_sockets
        if named_sockets is None:
            self.named_sockets = []
        self.seed = seed
        self.background = background
        self.instance_args = instance_args
        if instance_args is None:
            self.instance_args = [[] for _ in range(num_instances)]
        assert num_instances > 0
        assert len(self.instance_args) == num_instances

        self.blender_info = discover_blender(self.blend_path)
        if self.blender_info is None:
            logger.warning("Launching Blender failed;")
            raise ValueError("Blender not found or misconfigured.")
        else:
            logger.info(
                f'Blender found {self.blender_info["path"]} version {self.blender_info["major"]}.{self.blender_info["minor"]}'
            )

        self.launch_info = None
        self.processes = None

    def __enter__(self):
        """Launch processes"""
        assert self.launch_info is None, "Already launched."

        addresses = {}
        addrgen = self._address_generator(self.proto, self.bind_addr, self.start_port)
        for s in self.named_sockets:
            addresses[s] = [next(addrgen) for _ in range(self.num_instances)]

        seed = self.seed
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max - self.num_instances)
        seeds = [seed + i for i in range(self.num_instances)]

        instance_script_args = [[] for _ in range(self.num_instances)]
        for idx, iargs in enumerate(instance_script_args):
            iargs.extend(
                [
                    "-btid",
                    str(idx),
                    "-btseed",
                    str(seeds[idx]),
                    "-btsockets",
                ]
            )
            iargs.extend([f"{k}={v[idx]}" for k, v in addresses.items()])
            iargs.extend(self.instance_args[idx])

        popen_kwargs = {}
        if os.name == "posix":
            popen_kwargs = {"preexec_fn": os.setsid}
        elif os.name == "nt":
            popen_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

        processes = []
        commands = []
        env = os.environ.copy()
        for idx, script_args in enumerate(instance_script_args):
            cmd = [f'{self.blender_info["path"]}']
            if self.scene is not None and len(str(self.scene)) > 0:
                cmd.append(f"{self.scene}")
            if self.background:
                cmd.append("--background")
            cmd.append("--python-use-system-env")
            cmd.append("--enable-autoexec")
            cmd.append("--python")
            cmd.append(f"{self.script}")
            cmd.append("--")
            cmd.extend(script_args)

            p = subprocess.Popen(
                cmd,
                shell=False,
                stdin=None,
                stdout=None,
                stderr=None,
                env=env,
                **popen_kwargs,
            )

            processes.append(p)
            commands.append(" ".join(cmd))
            logger.info(f"Started instance: {cmd}")

        self.launch_info = LaunchInfo(addresses, commands, processes=processes)
        return self

    def assert_alive(self):
        """Tests if all launched process are alive."""
        if self.launch_info is None:
            return
        codes = self._poll()
        assert all([c is None for c in codes]), f"Alive test failed. Exit codes {codes}"

    def wait(self):
        """Wait until all launched processes terminate."""
        [p.wait() for p in self.launch_info.processes]

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Terminate all processes."""
        all_closed = all(
            [
                self._kill_tree(p.pid, sig=signal.SIGTERM, timeout=5.0)
                for p in self.launch_info.processes
            ]
        )
        if not all_closed:
            all_closed = all(
                [
                    self._kill_tree(p.pid, sig=signal.SIGKILL, timeout=5.0)
                    for p in self.launch_info.processes
                ]
            )
        self.launch_info = None
        if not all_closed:
            logger.warning("Not all Blender instances closed")
        else:
            logger.info("Blender instances closed")

    def _address_generator(self, proto, bind_addr, start_port):
        """Convenience to generate addresses."""
        if bind_addr == "primaryip":
            bind_addr = get_primary_ip()
        nextport = start_port
        while True:
            addr = f"{proto}://{bind_addr}:{nextport}"
            nextport += 1
            yield addr

    def _poll(self):
        """Convenience to poll all processes exit codes."""
        return [p.poll() for p in self.launch_info.processes]

    def _kill_tree(
        self,
        pid,
        sig=signal.SIGTERM,
        include_parent=True,
        timeout=None,
        on_terminate=None,
    ) -> bool:
        """Kill a process tree.

        This method is required for some tools actually spawn Blender as a subprocces (e.g. snap). This method
        ensures that the process opened and all its subprocesses are killed.
        """
        parent = psutil.Process(pid)
        plist = parent.children(recursive=True)
        if include_parent:
            plist.append(parent)

        for p in plist:
            p.send_signal(sig)

        gone, alive = psutil.wait_procs(plist, timeout=timeout, callback=on_terminate)

        return len(gone) == len(plist)


parent_pid = 30437  # my example
