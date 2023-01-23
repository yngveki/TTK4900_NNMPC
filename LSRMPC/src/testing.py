#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

class Reference:
    """
    Takes in a time associated with reference values that should be set for gas rate
    and oil rate specifically at that time.
    """
    
    def __init__(self, time=0, ref=None, nxt=None, prev=None):
        self.time = time
        
        assert ref != None, "\'ref\' must contain exactly two values (1 for gas rate, 1 for oil rate)"
        self.ref = ref
        self.nxt = nxt
        self.prev = prev

    def stripped(self):
        return [self.ref[0], self.ref[1]]

    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        return f"Reference object. (t = {self.time}, [{self.ref[0]}, {self.ref[1]}])"

    def __str__(self):
        return f"(t = {self.time}, [{self.ref[0]}, {self.ref[1]}])"

    def __getitem__(self, key):
        return self.ref[key]

class References:
    """
    Takes in a collection of reference values and their corresponding times,
    such that a given timestamp will result in an output reference value
    for both gas rate and oil rate.

    Example of input:

    refs = [(0, [11000,300]), (1000, [12000, 290])]
    """

    finished: bool

    # TODO: instead of taking in a list of refs, take in path to refs? fix everything internally?
    def __init__(self, refs, time=0):
        assert len(refs) > 0, "At least one reference must be given!"
        self.finished = False
        self.refs = refs#.sort()
        

        # Initialization of current reference
        self._curr_ref = self.refs[0]

        num_refs_left = len(self) - 1
        if num_refs_left > 0:
            self._link_refs()
        
        self._curr_time = time

    def _link_refs(self):
        self.refs[0].nxt = self.refs[1]

        for idx in range(1, len(self) - 1):
            self.refs[idx].prev = self.refs[idx - 1]
            self.refs[idx].nxt = self.refs[idx + 1]

        self.refs[-1].prev = self.refs[-2]

    @property
    def curr_ref(self):
        return self._curr_ref

    @curr_ref.setter
    def curr_ref(self, new_ref):
        self._curr_ref = new_ref
        
    @property
    def curr_time(self):
        return self._curr_time

    @curr_time.setter
    def curr_time(self, new_time):
        self._curr_time = new_time
        if self.curr_ref.nxt.time <= self.curr_time:
            self.curr_ref = self.curr_ref.nxt
            
    def __str__(self):
        out = ""
        for ref in self.refs:
            out += repr(ref) + "\n"

        return out

    def __len__(self):
        return len(self.refs)

if __name__ == "__main__":
    
    # Instantiate References-object
    ref_path = Path(__file__).parent / "../config/refs.csv"
    refs_frame = pd.read_csv(ref_path)
    refs = []
    for row in range(len(refs_frame)):
        ref = refs_frame.iloc[row,:]
        refs.append(Reference(time=ref[0],
                                ref=[ref[1],ref[2]]))
    refs = References(refs)
    print(refs.curr_ref)

    # Test switching of Reference within References-object upon iterating time
    print(refs.curr_ref)
    refs.curr_time = 3000
    print(refs.curr_ref)
    print(refs.curr_ref.stripped_print())