from pathlib import Path
import io
import pickle
import logging
import numpy as np
from pathlib import Path

_logger = logging.getLogger('blendtorch')   

class FileRecorder:
    def __init__(self, outpath='blendtorch.mpkl', max_messages=100000):
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        self.outpath = outpath
        self.capacity = max_messages
        _logger.info(f'Recording configured for path {outpath}, max_messages {max_messages}.')

    def save(self, data, is_pickled=False):
        if self.num_messages < self.capacity:
            offset = self.file.tell()
            if not is_pickled:
                self.pickler.dump(data)
            else:
                self.file.write(data)
            self.offsets[self.num_messages] = offset
            self.num_messages += 1

    def __enter__(self):
        self.file = io.open(self.outpath, 'wb')
        self.pickler = pickle.Pickler(self.file)        
        self.offsets = np.full(self.capacity, -1, dtype=int)
        self.num_messages = 0
        self.pickler.dump(self.offsets) # We fix this once we close the file.
        return self

    def __exit__(self, *args):
        self.file.seek(0)
        # need to re-create unpickler
        pickle.Pickler(self.file).dump(self.offsets)
        self.file.close()
        self.file = None

    @staticmethod
    def filename(prefix, worker_idx):
        return f'{prefix}_{worker_idx:02d}.btr'

class FileReader:
    def __init__(self, path):
        self.path = path
        self.offsets = FileReader.read_offsets(path) 
        self._file = None

    def __len__(self):
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
        self._file = io.open(self.path, 'rb')
        self._unpickler = pickle.Unpickler(self._file)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    @staticmethod
    def read_offsets(fname):
        assert Path(fname).exists(), f'Cannot open {fname} for reading.'
        with io.open(fname, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            offsets = unpickler.load()
            num_messages = len(offsets)

            m = np.where(offsets==-1)[0]            
            if len(m) > 0:
                num_messages = m[0]
            return offsets[:num_messages]