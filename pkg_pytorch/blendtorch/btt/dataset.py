import zmq
import pickle
from contextlib import ExitStack
from glob import glob
import torch.utils as utils

from .file import FileRecorder, FileReader
from .constants import DEFAULT_TIMEOUTMS


def _identity_item_transform(x):
    return x

class RemoteIterableDataset(utils.data.IterableDataset):
    '''Base class for iteratable datasets that stream from remote Blender instances.'''

    def __init__(self, addresses, queue_size=10, timeoutms=DEFAULT_TIMEOUTMS, max_items=100000, item_transform=None, record_path_prefix=None):
        self.addresses = addresses
        self.queue_size = queue_size
        self.timeoutms = timeoutms
        self.max_items = max_items
        self.record_path_prefix = record_path_prefix
        self.item_transform = item_transform or _identity_item_transform

    def enable_recording(self, fname):
        self.record_path_prefix = fname

    def stream_length(self, max_items):
        self.max_items = max_items

    def __iter__(self):
        return self._stream()

    def _stream(self):
        ctx = zmq.Context()
        socket = None

        try:
            socket = ctx.socket(zmq.PULL)
            socket.setsockopt(zmq.RCVHWM, self.queue_size)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            for addr in self.addresses:
                socket.connect(addr)

            num_workers = 1
            worker_id = 0
            wi = utils.data.get_worker_info()
            if wi is not None:
                worker_id = wi.id
                num_workers = wi.num_workers
            
            with ExitStack() as es:
                rec = None                
                if self.record_path_prefix is not None:
                    rec = es.enter_context(
                        FileRecorder(
                            FileRecorder.filename(self.record_path_prefix, worker_id),
                            self.max_items
                        )
                    )
                

                for i in range(self.max_items//num_workers):
                    socks = dict(poller.poll(self.timeoutms))
                    assert socket in socks, 'No response within timeout interval.'
                    if rec:
                        data = socket.recv()
                        rec.save(data, is_pickled=True)
                        obj = pickle.loads(data)
                    else:
                        obj = socket.recv_pyobj()
                    yield self._item(obj)
            
        finally:
            if socket is not None:
                socket.close()

    def _item(self, item):
        return self.item_transform(item)

class SingleFileDataset(utils.data.Dataset):
    '''Replays from a specific recording file.'''
    def __init__(self, path, item_transform=None):
        self.reader = FileReader(path)
        self.item_transform = item_transform or _identity_item_transform

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        return self._item(self.reader[idx])

    def _item(self, item):
        return self.item_transform(item)

class FileDataset(utils.data.ConcatDataset):
    '''Replays from multiple recording files matching a recording pattern.'''
    def __init__(self, record_path_prefix, item_transform=None):
        fnames = sorted(glob(f'{record_path_prefix}_*.btr'))
        assert len(fnames) > 0, f'Found no recording files with prefix {record_path_prefix}'
        ds = [SingleFileDataset(fname) for fname in fnames]
        super().__init__(ds)

        self.item_transform = item_transform or _identity_item_transform

    def __getitem__(self, idx):
        return self._item(super().__getitem__(idx))  

    def _item(self, item):
        return self.item_transform(item)