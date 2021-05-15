import numpy as np, os
from base_dsl import BaseDSL, one_hot_labels
from os.path import expanduser, join
from cfg import cfg

class ImagenetDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=True, mode='train', val_frac=None, normalize_channels=False, path=None, resize=None, start_batch=1, end_batch=10, num_to_keep=None):

        assert val_frac is None, 'This dataset has pre-specified splits.'
        assert start_batch >= 1
        assert end_batch <= 10
        
        self.start_batch = start_batch
        self.end_batch = end_batch

        if path is None:
            self.path = os.path.join('datasets', 'Imagenet64')
            
            # Uncomment to use run.py
            self.path = os.path.join(cfg.home, 'datasets', 'Imagenet64')
        else:
            self.path = path
        
        self.num_to_keep = num_to_keep
        
        super(ImagenetDSL, self).__init__(
            batch_size,
            shuffle_each_epoch=shuffle_each_epoch,
            seed=seed,
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        xs = []
        ys = []

        if mode == 'train':
            data_files = [os.path.join(self.path, 'train_data_batch_%d.npy' % idx) for idx in range(self.start_batch, self.end_batch+1)]
        else:
            assert mode == 'val', 'Mode not supported.'
            data_files = [os.path.join(self.path, 'val_data.npy')]

        for data_file in data_files:
            print(data_file)
            d = np.load(data_file,allow_pickle=True).item()
            x = np.array(d['data'])
            y = np.array(d['labels'])
            
            y = [i-1 for i in y]

            img_size  = 64
            img_size2 = img_size * img_size

            x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
            x = x.reshape((x.shape[0], img_size, img_size, 3))

            xs.append(x)
            ys.append(np.array(y))

        if len(xs) == 1:
            self.data   = xs[0]
            self.labels = ys[0]
        else:
            self.data   = np.concatenate(xs, axis=0)
            self.labels = np.concatenate(ys, axis=0)

        if self.num_to_keep is not None:
            self.shuffle_data()
            self.data = self.data[:self.num_to_keep]
            self.labels = self.labels[:self.num_to_keep]

    def convert_Y(self, Y):
        return one_hot_labels(Y, 1000)
