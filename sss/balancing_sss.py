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

class BalancingSelectionStrategy(SubsetSelectionStrategy):
    def __init__(self, size, Y_idx, num_stacks, idx=None):
        self.num_stacks = num_stacks
        self.Y_idx      = Y_idx
        self.idx        = idx
        super(BalancingSelectionStrategy, self).__init__(size, Y_vec=None)

    def get_subset(self):
        s = [ [] for i in range(self.num_stacks) ]
        
        q = self.size
        
        if self.idx is not None:
            for i in self.idx:
                s[self.Y_idx[i]].append(i)        
        else:
            for i in range(self.Y_idx.shape[0]):
                s[self.Y_idx[i]].append(i)

        for i in range(self.num_stacks):
            print "stack-{} len: {}" .format(i, len(s[i]) )

        subset = []

        while(q):
            for i in range(self.num_stacks):
                if len(s[i]) != 0:
                    subset.append( s[i].pop() )
                    q = q-1

                if q == 0:
                    break

        return subset