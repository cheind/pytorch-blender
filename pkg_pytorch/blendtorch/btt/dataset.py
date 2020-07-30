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
    '''Base class for iteratable datasets that receive data from remote Blender instances.

    To support multiple DataLoader workers in PyTorch, this class lazily constructs the data
    stream upon start of iteration. Each received message is represented as item dictionary.

    `RemoteIterableDataset` supports two ways to manipulate items before returning them to
    the caller
     - Provide an `item_transform` that takes a dictionary and returns transformed elements
     - Inherit from `RemoteIterableDataset` and override `RemoteIterableDataset._item()` method.
    
    Params
    ------
    addresses: list-like
        ZMQ addresses to connect to.
    max_items: integer
        Artificial length of this dataset. Also affects the 
        maximum capacity of any recorder used to record messages.
    item_transform: callable
        Any transform to apply to received items. Each item is
        a dictionary whose content is defined by the Blender script
        sending the data via `btb.DataPublisher`.
    record_path_prefix: str, Path
        Path prefix to record to. When given, each DataLoader worker will
        create a recording file `{prefix}_{worker_idx:02d}.btr`.
    queue_size: integer
        Receive queue size before publisher get stalled.
    timeoutms: integer
        Max wait time before raising an error.
    '''

    def __init__(self, addresses, queue_size=10, timeoutms=DEFAULT_TIMEOUTMS, max_items=100000, item_transform=None, record_path_prefix=None):
        self.addresses = addresses
        self.queue_size = queue_size
        self.timeoutms = timeoutms
        self.max_items = max_items
        self.record_path_prefix = record_path_prefix
        self.item_transform = item_transform or _identity_item_transform

    def enable_recording(self, fname):
        '''Enable recording to given prefix path `fname`.
        
        Needs to be set before receiving items from the dataset.
        
        '''
        self.record_path_prefix = fname

    def stream_length(self, max_items):
        '''Return artificial dataset length.'''
        self.max_items = max_items

    def __iter__(self):
        '''Return a dataset iterator.'''
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
        '''Transform the given item.
        Defaults to applying the `item_transform`.
        '''
        return self.item_transform(item)

class SingleFileDataset(utils.data.Dataset):
    '''Replays from a particular recording file.'''
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
    '''Replays from multiple recordings matching a recording pattern.
    
    This dataset constructs one `SingleFileDataset` per file matching
    the specified prefix pattern `record_path_prefix`. All datasets
    are then concatenated to appear as a single larger dataset.     
    '''
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