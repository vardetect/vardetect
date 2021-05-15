"""
MIT License

Copyright (c) 2019 Soham Pal, Yash Gupta, Aditya Kanade, Shirish Shevade, Vinod Ganapathy. Indian Institute of Science.
Modified in 2019 by Yash Gupta, Soham Pal, Aditya Kanade, Shirish Shevade. Indian Institute of Science.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from base_dsl import BaseDSL
import random

def collect_aux_data(dictionary_list, key):
    result = []
    
    for item in dictionary_list:
        result.append(item[key])
            
    return np.array(result)

class DSLMarker:
    def __init__(self, orig_dsl):
        assert isinstance(orig_dsl, BaseDSL)
        assert orig_dsl.shuffle_each_epoch == False
        
        # Initialize hashsets and hashmap
        self.marked = []
        self.unmarked = []
        self.marked_set = set()
        
        self.aux_data = {}
        self.orig_dsl = orig_dsl
        self.marked_pos_index = 0
        self.unmarked_pos_index = 0
        self._marked_counter = 0
        self._unmarked_counter = 0
        
        self.batches = orig_dsl.get_num_batches()
        self.batch_size = orig_dsl.get_batch_size()
        self.num_samples = orig_dsl.get_num_samples()
        
        for i in range(0, self.num_samples):
                self.unmarked.append(i)
                
    def shuffle_data(self):
#         random.setstate(self.orig_dsl.random_state)

#         perm = np.arange(self.orig_dsl.data.shape[0]) 
#         random.shuffle(perm)
#         self.orig_dsl.data   = self.orig_dsl.data[perm]
        
#         if self.orig_dsl.labels is not None:
#             self.orig_dsl.labels = self.orig_dsl.labels[perm]
        
#         self.marked = [perm[i] for i in self.marked]
#         self.unmarked = [perm[i] for i in self.unmarked]
#         self.marked_set = set(self.marked)
        
        random.setstate(self.orig_dsl.random_state)
        random.shuffle(self.marked)
        random.shuffle(self.unmarked)
    
        self.marked_pos_index = 0
        self.unmarked_pos_index = 0
        self.orig_dsl.random_state = random.getstate()

#         self.orig_dsl.random_state = random.getstate()
                
    def get_split_dsls(self):
         class DSLWrapper(BaseDSL):
            def __init__(self, dsl_marker, marked):
                self.dsl_marker = dsl_marker
                self.is_multilabel = dsl_marker.orig_dsl.is_multilabel
                
                if marked:
                    self.load_next_batch = dsl_marker.load_next_marked_batch
                else:
                    self.load_next_batch = dsl_marker.load_next_unmarked_batch
                    
                self.convert_Y        = dsl_marker.orig_dsl.convert_Y
                self.get_num_classes  = dsl_marker.orig_dsl.get_num_classes
                self.get_sample_shape = dsl_marker.orig_dsl.get_sample_shape
                self.get_batch_size   = dsl_marker.orig_dsl.get_batch_size
                
                if marked:
                    self.get_num_samples = lambda : len(self.dsl_marker.marked)
                else:
                    self.get_num_samples = lambda : len(self.dsl_marker.unmarked)
                
                self.get_num_batches = lambda : np.ceil(
                                                    float(self.get_num_samples())/
                                                    self.get_batch_size()
                                                ).astype(np.int32)
        
         return DSLWrapper(self, True), DSLWrapper(self, False)
    
    def mark(self, i, aux_data=None):
        assert i not in self.marked_set
        
        if i not in self.marked_set:
            self.unmarked.remove(i)
            self.marked_set.add(i)
            self.marked.append(i)
        
        self.aux_data[i] = aux_data
        
    def update(self, i, aux_data=None):
        assert i in self.marked_set
        
        self.aux_data[i] = aux_data
        
    def update_unmarked_pointer(self, initial=False):
        wrapped = False
        
        if not initial:
            self._unmarked_counter += 1
            
            assert self._unmarked_counter <= len(self.unmarked)
            
            if self._unmarked_counter >= len(self.unmarked):
                wrapped = True
                self._unmarked_counter = 0
        
        self.unmarked_pos_index = self.unmarked[self._unmarked_counter]
        
        return wrapped
        
    def update_marked_pointer(self, initial=False):
        wrapped = False
        
        if not initial:
            self._marked_counter += 1
            
            assert self._marked_counter <= len(self.marked)
            
            if self._marked_counter >= len(self.marked):
                wrapped = True
                self._marked_counter = 0
        
        self.marked_pos_index = self.marked[self._marked_counter]
        
        return wrapped
    
    def load_next_marked_batch(self, batch_num=None, return_idx=False, return_aux=False):
        assert len(self.marked) > 0
        assert batch_num is None
        
        X = np.empty(tuple([self.batch_size] + list(self.orig_dsl.data[0].shape)), dtype=self.orig_dsl.data.dtype)
        
        if self.orig_dsl.labels is not None:
            Y = np.empty(tuple([self.batch_size] + list(self.orig_dsl.labels[0].shape)))
        else:
            Y = None
        
        idx = np.empty(tuple([self.batch_size]),dtype=np.int32)
        aux = []
        
        broke = False
        
        for i in range(self.batch_size):
            wrapped = self.update_marked_pointer(initial=i==0)
            
            if wrapped and i != 0:
                broke = True
                i -= 1
                break
                       
            X[i] = self.orig_dsl.data[self.marked_pos_index]
            
            if Y is not None:
                Y[i] = self.orig_dsl.labels[self.marked_pos_index]
            
            if return_aux:
                aux.append(self.aux_data[self.marked_pos_index])
            
            idx[i] = self.marked_pos_index
        
        X = X[:i+1]
        
        if Y is not None:
            Y = Y[:i+1]
        
        idx = idx[:i+1]
        
        if return_aux:
            aux = np.array(aux)
        
        if not broke:
            self.update_marked_pointer()
        
        if return_idx and return_aux:
            return X, Y, aux, idx
        elif return_idx and not return_aux:
            return X, Y, idx
        elif not return_idx and return_aux:
            return X, Y, aux
        else:
            return X, Y
    
    def load_next_unmarked_batch(self, batch_num=None, return_idx=False):
        assert len(self.unmarked) > 0
        assert batch_num is None
        
        X = np.empty(tuple([self.batch_size] + list(self.orig_dsl.data[0].shape)))
        
        if self.orig_dsl.labels is not None:
            Y = np.empty(tuple([self.batch_size] + list(self.orig_dsl.labels[0].shape)))
        else:
            Y = None
        
        idx = np.empty(tuple([self.batch_size]),dtype=np.int32)
        
        broke = False
        
        for i in range(self.batch_size):
            wrapped = self.update_unmarked_pointer(initial=i==0)
            
            if wrapped and i != 0:
                broke = True
                i -= 1
                break
            
            X[i] = self.orig_dsl.data[self.unmarked_pos_index]
            
            if Y is not None:
                Y[i] = self.orig_dsl.labels[self.unmarked_pos_index]
            
            idx[i] = self.unmarked_pos_index
        
        X = X[:i+1]
        
        if Y is not None:
            Y = Y[:i+1]
        
        idx = idx[:i+1]
        
        if not broke:
            self.update_unmarked_pointer()
        
        if return_idx:
            return X, Y, idx
        else:
            return X, Y
