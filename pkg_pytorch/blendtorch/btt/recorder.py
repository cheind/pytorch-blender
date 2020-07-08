from pathlib import Path
import io
import pickle
import numpy as np

class Recorder:
    def __init__(self, outpath='blendtorch.mpkl', max_messages=1000):
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        self.file = io.open(outpath, 'wb')
        self.pickler = pickle.Pickler(self.file)        
        self.offsets = np.full(max_messages, -1, dtype=int)
        self.num_messages = 0
        self.max_messages = max_messages
        self.pickler.dump(self.offsets) # We fix this once we close the file.
        # https://gist.github.com/marekyggdrasil/14bbb9e2ca91f9a1fad8495b5ac59354

    def save(self, data, is_pickled=False):
        if self.num_messages < self.max_messages:
            offset = self.file.tell()
            if not is_pickled:
                self.pickler.dump(data)
            else:
                self.file.write(data)
            self.offsets[self.num_messages] = offset
            self.num_messages += 1

    def close(self):
        self.file.seek(0)
        # need to re-create unpickler
        pickle.Pickler(self.file).dump(self.offsets)
        self.file.close()
        self.file = None


