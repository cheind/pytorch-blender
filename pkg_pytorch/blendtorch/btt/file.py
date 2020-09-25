from pathlib import Path
import io
import pickle
import logging
import numpy as np
from pathlib import Path

_logger = logging.getLogger('blendtorch')   

class FileRecorder:
    '''Provides raw message recording functionality.
    
    `FileRecorder` stores received pickled messages into a single file.
    This file consists of a header and a sequence of messages received.
    Upon closing, the header is updated to match the offsets of pickled
    messages in the file. The offsets allow `FileReader` to quickly
    read and unpickle a specific message stored in this file.

    This class is meant to be used as context manager.

    Params
    ------
    outpath: str
        File path to write to
    max_messages: int
        Only up to `max_messages` can be stored.
    '''
    def __init__(self, outpath='blendtorch.mpkl', max_messages=100000, update_header=10000):
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        self.outpath = outpath
        self.capacity = max_messages
        self.update_header = update_header
        _logger.info(f'Recording configured for path {outpath}, max_messages {max_messages}.')

    def save(self, data, is_pickled=False):
        '''Save new data if there is still capacity left.

        Params
        ------
        data : object
            Pickled bytes or unpickled message.
        is_pickled: bool
            Whether or not the input is already in pickled
            represenation.
        '''
        if self.num_messages < self.capacity:
            offset = self.file.tell()
            self.offsets[self.num_messages] = offset
            self.num_messages += 1
            if not is_pickled:
                self.pickler.dump(data)
            else:
                self.file.write(data)
        
        if self.num_messages % self.update_header == 0:
            self._update_offsets()
            

    def __enter__(self):
        self.file = io.open(self.outpath, 'wb', buffering=0)
        # We currently cannot use the highest protocol but use a version
        # compatible with Python 3.7 (Blender version).
        # TODO even if we set this to highest (-1) and only run the tests
        # we get an error when loading data from stream. We should ensure
        # that what we do here is actually OK, independent of the protocol.
        self.pickler = pickle.Pickler(self.file, protocol=3) 
        self.offsets = np.full(self.capacity, -1, dtype=np.int64)
        self.num_messages = 0
        self.pickler.dump(self.offsets) # We fix this once we update headers.
        return self

    def __exit__(self, *args):
        self._update_offsets()
        self.file.close()
        self.file = None

    def _update_offsets(self):
        off = self.file.tell()
        self.file.seek(0)
        pickle.Pickler(self.file, protocol=3).dump(self.offsets)
        self.file.seek(off)


    @staticmethod
    def filename(prefix, worker_idx):
        '''Return a unique filename for the given prefix and worker id.'''
        return f'{prefix}_{worker_idx:02d}.btr'

class FileReader:
    '''Read items from file.

    Assumes file was written by `FileRecorder`.

    Params
    ------
    path: str
        File path to read.    
    '''
    def __init__(self, path):
        self.path = path
        self.offsets = FileReader.read_offsets(path) 
        self._file = None

    def __len__(self):
        '''Returns number of items stored.'''
        return len(self.offsets)

    def __getitem__(self, idx):
        '''Read message associated with index `idx`.'''
        if self._file is None:
            # Lazly creating file object here to ensure PyTorch 
            # multiprocessing compatibility (i.e num_workers > 0)
            self._create()
        
        self._file.seek(self.offsets[idx])
        return self._unpickler.load()

    def _create(self):
        self._file = io.open(self.path, 'rb', buffering=0)
        self._unpickler = pickle.Unpickler(self._file)

    def close(self):
        '''Close the reader.'''
        if self._file is not None:
            self._file.close()
            self._file = None

    @staticmethod
    def read_offsets(fname):
        '''Returns the offset header of file refered to by `fname`.'''
        assert Path(fname).exists(), f'Cannot open {fname} for reading.'
        with io.open(fname, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            offsets = unpickler.load()
            num_messages = len(offsets)

            m = np.where(offsets==-1)[0]            
            if len(m) > 0:
                num_messages = m[0]
            return offsets[:num_messages]