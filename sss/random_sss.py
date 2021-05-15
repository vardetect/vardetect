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

from base_sss import SubsetSelectionStrategy
import base_sss
import random

class RandomSelectionStrategy(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec):
        super(RandomSelectionStrategy, self).__init__(size, Y_vec)
        
    
    def get_subset(self):
        random.setstate(base_sss.sss_random_state)
        s = random.sample([i for i in range(len(self.Y_vec))],  self.size)
        base_sss.sss_random_state = random.getstate()
        return s