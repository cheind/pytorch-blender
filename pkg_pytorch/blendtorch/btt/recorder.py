from pathlib import Path
import io
import pickle
import logging
import numpy as np

_logger = logging.getLogger('blendtorch')

class Recorder:
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

