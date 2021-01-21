import numpy as np
import random
from base_dsl import BaseDSL

class UniformDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=False, mode='train', val_frac=None, normalize_channels=False, resize=None, shape=None, sample_limit=None):
        assert resize is None, 'Does not support resizing.'
        assert shape is not None, 'Shape must be specified.'
        assert normalize is False, 'Normalization is not supported.'
        assert normalize_channels is False, 'Normalization is not supported.'
        assert shuffle_each_epoch is False, 'Shuffle_each_epoch is not supported.'
        
        assert sample_limit is not None, "Deprecated"
        
        self.shuffle_each_epoch = False
        
        self.seed        = seed
        self.batch_size  = batch_size
        self.labels      = None
        
        if mode == 'val':
            self.seed *= 2
        elif mode == 'test':
            self.seed *= 3
        
        if sample_limit is None:
            assert shuffle_each_epoch is False
            self.num_batches = np.infty
            self.num_samples = np.infty
            
            self.get_num_batches = lambda : np.infty
            self.get_num_samples = lambda : np.infty
        else:
            self.num_batches = int(np.ceil(sample_limit/float(batch_size)))
            self.num_samples = sample_limit
        
        self.batch_index = 0
        
        self.shape = shape
        self.load_data(mode, val_frac)
        self.random_state = random.getstate()
    
    def load_data(self, mode, val_frac):
        if np.isinf(self.num_samples):
            self.load_next_batch = self._load_next_batch
        else:
            self.data = np.empty((self.num_samples, self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
            self.labels = None
            self.rs = np.random.RandomState(self.seed)
            
            for i in range(self.num_batches):
                if i == self.num_batches - 1:
                    bs  = self.num_samples - ( i * self.batch_size )
                else:
                    bs = self.batch_size
                
                self.data[i*self.batch_size:(i+1)*self.batch_size] = self.rs.uniform(size=(bs, self.shape[0], self.shape[1], self.shape[2]))

    def get_sample_shape(self):
        return self.shape

    def _load_next_batch(self, batch_index):
        if batch_index == 0:
            self.rs = np.random.RandomState(self.seed)
        
        X = self.rs.uniform(size=(self.batch_size, self.shape[0], self.shape[1], self.shape[2]))
        
        return X, None
