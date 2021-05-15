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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re, nltk
import numpy as np
import six
import collections
import six, copy
from collections import OrderedDict 
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated
from nltk.corpus import stopwords

# Usage:
# -----
# text = ["hello yash", "hi karan hello is yash"]
# vocab_processor = VocabularyProcessor(max_document_length=3, min_frequency=1, max_frequency=-1)
# x = np.array(list(vocab_processor.fit_transform(text)))
# vocab_dict = vocab_processor.vocabulary_._mapping

try:
  # pylint: disable=g-import-not-at-top
  import cPickle as pickle
except ImportError:
  # pylint: disable=g-import-not-at-top
  import pickle

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",re.UNICODE)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def stop_word_removal(text):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = pattern.sub('', text)
    return text

@deprecated(None, 'Please use tensorflow/transform or tf.data.')
def tokenizer(iterator, tokenizer_fn):
    """Tokenizer generator.
    Args:
    iterator: Input iterator with strings.
    Yields:
    array of tokens per each value in the input.
    """
    for value in iterator:
        cleanstr           = cleanhtml(value)
        yield tokenizer_fn(cleanstr)


class CategoricalVocabulary(object):
    """Categorical variables vocabulary class.
    THIS CLASS IS DEPRECATED. See
    [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
    for general migration instructions.
    Accumulates and provides mapping from classes to indexes.
    Can be easily used for words.
    """
    
    def __init__(self, unknown_token="<unk>", support_reverse=True, pad_token="<pad>", start_token="<start>", end_token="<end>"  ):
        self._unknown_token = unknown_token
        self._pad_token     = pad_token
        self._start_token   = start_token
        self._end_token     = end_token
        self._keywords      = [ pad_token, unknown_token, start_token, end_token ]
            
        self._total_tokens  = None    
            
        self._mapping = { pad_token : 0, unknown_token: 1, start_token: 2,  end_token: 3 }
        self._support_reverse = support_reverse
        if support_reverse:
            self._reverse_mapping = [pad_token, unknown_token, start_token, end_token ]
            
        self._freq = collections.defaultdict(int)
        self._freeze = False

    def __len__(self):
        """Returns total count of mappings. Including unknown token."""
        return len(self._mapping)

    def get_vocabulary_size( self ):
        return self.__len__()
    
    def freeze(self, freeze=True):
        """Freezes the vocabulary, after which new words return unknown token id.
        Args:
          freeze: True to freeze, False to unfreeze.
        """
        self._freeze = freeze

    def get(self, category):
        """Returns word's id in the vocabulary.
        If category is new, creates a new id for it.
        Args:
          category: string or integer to lookup in vocabulary.
        Returns:
          interger, id in the vocabulary.
        """
        if category in self._keywords:
            return self._mapping[category]        
        elif category not in self._mapping:
            if self._freeze:
                return self._mapping[self._unknown_token]
            self._mapping[category] = len(self._mapping)
            if self._support_reverse:
                self._reverse_mapping.append(category)

        return self._mapping[category]

    def add(self, category, count=1):
        """Adds count of the category to the frequency table.
        Args:
          category: string or integer, category to add frequency to.
          count: optional integer, how many to add.
        """
        category_id = self.get(category)
        if category_id <= 3:
            return
        self._freq[category] += count

    def trim(self, min_frequency, max_frequency=-1, num_words=None):
        """Trims vocabulary for minimum frequency.
        Remaps ids from 1..n in sort frequency order.
        where n - number of elements left.
        Args:
          min_frequency: minimum frequency to keep.
          max_frequency: optional, maximum frequency to keep.
            Useful to remove very frequent categories (like stop words).
        """
        
        print("self._freq.items()", len( self._freq ) )
        
        self._trunc_freq = sorted(self._freq.items() ,  key=lambda x: x[1] , reverse = True )
        
        self._mapping = { self._pad_token : 0, self._unknown_token: 1, self._start_token: 2, self._end_token: 3  }
        
        if self._support_reverse:
             self._reverse_mapping = [self._pad_token, self._unknown_token, self._start_token, self._end_token ]
        
        idx = 4
                
        print("self _trunc_freq ", len(self._trunc_freq) )
            
        if min_frequency > 1:
            print("Keeping words which have minimum of {} frequency" .format( min_frequency ) )
                  
        if max_frequency > -1:
            print("Keeping words which have maximum of {} frequency" .format( max_frequency )  )                
        
        if num_words is not None:
            print("Truncating vocabulary size to {} words" .format(num_words) )

        for category, count in self._trunc_freq:                                                
            if max_frequency > 0 and count >= max_frequency:
                print("Removing token '{}' with frequency {} which exceeds max frequency" .format( category, count ) )
                continue
            if count < min_frequency:
                break
            self._mapping[category] = idx
            idx += 1
            if self._support_reverse:
                self._reverse_mapping.append(category)
            assert len(self._mapping) <= num_words
            if num_words is not None and len(self._mapping) == num_words:
                break
                
        self._trunc_freq = OrderedDict(self._trunc_freq[:idx - 1])

    def reverse(self, class_id):
        """Given class id reverse to original class name.
        Args:
          class_id: Id of the class.
        Returns:
          Class name.
        Raises:
          ValueError: if this vocabulary wasn't initialized with support_reverse.
        """
        if not self._support_reverse:
            raise ValueError("This vocabulary wasn't initialized with support_reverse to support reverse() function.")
        return self._reverse_mapping[class_id]



class VocabularyProcessor(object):
    """Maps documents to sequences of word ids.
    THIS CLASS IS DEPRECATED. See
    [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
    for general migration instructions.
    """

    def __init__(self, max_document_length, min_frequency=1, max_frequency=-1, vocabulary=None, tokenizer_fn = None, num_words=None, padding_loc='right'):
        """Initializes a VocabularyProcessor instance.
        Args:
          max_document_length: Maximum length of documents.
            if documents are longer, they will be trimmed, if shorter - padded.
          min_frequency: Minimum frequency of words in the vocabulary.
          vocabulary: CategoricalVocabulary object.
        Attributes:
          vocabulary_: CategoricalVocabulary object.
        """
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.num_words = num_words
        self.padding_loc = padding_loc
        if vocabulary:
            self.vocabulary_ = vocabulary
        else:
            self.vocabulary_ = CategoricalVocabulary()
        if tokenizer_fn is not None:
            self._tokenizer = tokenizer_fn
        else:
            self._tokenizer = TOKENIZER_RE.findall

    def fit(self, raw_documents, unused_y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.
        Args:
          raw_documents: An iterable which yield either str or unicode.
          unused_y: to match fit format signature of estimators.
        Returns:
          self
        """
       
        stop_words = set(stopwords.words('english'))         
        self.token_lengths = []
        for tokens in tokenizer(raw_documents, self._tokenizer):            
            self.token_lengths.append( len(tokens) )
            for token in tokens:
                self.vocabulary_.add(token.lower())
        
        print("MaxLen: {}  MinLen: {} MeanLen: {}" .format( np.max(self.token_lengths), np.min(self.token_lengths), np.mean(self.token_lengths) ) )           
        print("Vocabulary size before trimming: {}"  .format( len(self.vocabulary_._mapping) ) )
            
        orderedTuples           = sorted(self.vocabulary_._freq.items() ,  key=lambda x: x[1] , reverse = True )
        self.vocabulary_._freq  =  OrderedDict( orderedTuples )
                        
        if self.min_frequency > 1 or self.max_frequency > -1 or self.num_words is not None:
            self.vocabulary_.trim(self.min_frequency, self.max_frequency, self.num_words)
        else:
            self.vocabulary_._trunc_freq = self.vocabulary_._freq
            
        self.vocabulary_.freeze()
        
        print("Vocabulary size after trimming: {}"  .format( len(self.vocabulary_._mapping) ) )
        
        total_tokens   = sum( self.vocabulary_._trunc_freq.values() )
        
        self.vocabulary_.total_tokens = total_tokens
        
        print("total tokens in vocal ", total_tokens)
        
        return self

    def fit_transform(self, raw_documents, lengths=None, unused_y=None):
        """Learn the vocabulary dictionary and return indexies of words.
        Args:
          raw_documents: An iterable which yield either str or unicode.
          unused_y: to match fit_transform signature of estimators.
        Returns:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        self.fit(raw_documents)
        return self.transform(raw_documents, lengths)

    def transform(self, raw_documents, lengths):
        """Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Args:
          raw_documents: An iterable which yield either str or unicode.
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        
        for tokens in tokenizer(raw_documents, self._tokenizer):
            word_ids = np.zeros(self.max_document_length, np.int64)
            if self.padding_loc == 'left':
                idx = max(0, self.max_document_length - ( len(tokens) + 2 ) )
            else:
                idx = 0
            
            word_ids[idx] = self.vocabulary_.get(self.vocabulary_._start_token)            
            idx = idx + 1
            
            if lengths is not None:
                lengths.append( min( self.max_document_length, len(tokens)+2 )  )
            
            for token in tokens:
                if idx >= (self.max_document_length-1):
                    break
                word_ids[idx] = self.vocabulary_.get(token.lower())
                idx = idx + 1
                
            word_ids[idx] = self.vocabulary_.get(self.vocabulary_._end_token)
            
            yield word_ids

    def reverse(self, documents):
        """Reverses output of vocabulary mapping to words.
        Args:
          documents: iterable, list of class ids.
        Yields:
          Iterator over mapped in words documents.
        """
        for item in documents:
            output = []
            for class_id in item:
                output.append(self.vocabulary_.reverse(class_id))
            yield ' '.join(output)

    def save(self, filename):
        """Saves vocabulary processor into given file.
        Args:
          filename: Path to output file.
        """
        with gfile.Open(filename, 'wb') as f:
            f.write(pickle.dumps(self))

    @classmethod
    def restore(cls, filename):
        """Restores vocabulary processor from given file.
        Args:
          filename: Path to file to load from.
        Returns:
          VocabularyProcessor object.
        """
        with gfile.Open(filename, 'rb') as f:
            return pickle.loads(f.read())
