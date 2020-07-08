import pickle
import io
import numpy as np

class ReplayDataset:
    def __init__(self, record_path):
        self.file = io.open(record_path, 'rb')
        self.unpickler = pickle.Unpickler(self.file)
        self.offsets = self.unpickler.load()
        print(self.offsets)
        m = np.where(self.offsets==-1)[0]
        if len(m)==0:
            self.num_messages = len(self.offsets)
        else:
            self.num_messages = m[0]        
        
    def __len__(self):
        return self.num_messages

    def __getitem__(self, idx):        
        self.file.seek(self.offsets[idx])
        return self.unpickler.load()

if __name__ == '__main__':
    ds = ReplayDataset('record.mpkl')
    print(len(ds))
