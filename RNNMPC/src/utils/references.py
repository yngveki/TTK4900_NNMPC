#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

class Reference:
    """
    Takes in a time associated with reference values that should be set for gas rate
    and oil rate specifically at that time.
    """
    
    def __init__(self, time=0, ref=None, nxt=None, prev=None):
        self.time = time
        # self._time = time
        
        assert ref != None, "\'ref\' must contain exactly two values (1 for gas rate, 1 for oil rate)"
        self.ref = ref
        self.nxt = nxt
        self.prev = prev

    # @property
    # def time(self):
    #     return self._time

    # @time.setter
    # def time(self, new_time):
    #     self._time = new_time

    def stripped(self):
        return [self.ref[0], self.ref[1]]

    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        return f"Reference object. (t = {self.time}, [{self.ref[0]}, {self.ref[1]}]), [prev={self.prev}, nxt={self.nxt}]"

    def __str__(self):
        return f"(t = {self.time}, [{self.ref[0]}, {self.ref[1]}])"

    def __getitem__(self, key):
        return self.ref[key]

    def __len__(self):
        return len(self.ref)

class References:
    """
    Takes in a collection of reference values and their corresponding times,
    such that a given timestamp will result in an output reference value
    for both gas rate and oil rate.

    Example of input:

    refs = [(0, [11000,300]), (1000, [12000, 290])]

    Note that this implementation only gives out a single reference that is the
    currently valid one, based on time. If the references are to be projected
    over some time, and may vary during that time if a new reference becomes valid
    during that time into the future, the class ReferencesTimeseries must be used.
    """

    def __init__(self, ref_path, time=0):
        refs_frame = pd.read_csv(ref_path)
        assert len(refs_frame) > 0, "At least one reference must be given!"

        refs = []
        for row in range(len(refs_frame)):
            ref = refs_frame.iloc[row,:]
            refs.append(Reference(time=ref[0],
                                  ref=[ref[1],ref[2]]))
        refs.sort() # In case given references are not sorted chronologically
        self.refs = refs
        
        # Initialization of current reference
        self._curr_ref = self.refs[0]

        num_refs_left = len(self) - 1
        if num_refs_left > 0:
            self._link_refs()
        
        self._curr_time = time

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
        if self.curr_ref.nxt != None:
            if self.curr_ref.nxt.time <= self.curr_time:
                self.curr_ref = self.curr_ref.nxt

    def refs_as_lists(self):
        """
        Turns the Reference objects within self.refs into lists,
        discarding the corresponding timestamps.
        
        Ex.: [(t=0,[100,0]), (t=4,[90,10])] -> [[100,0],[90,10]]
        """

        # Knowing future reference changes is unrealistic; always assume static reference
        return [self[0].stripped() for _ in range(len(self))]
        

    def _link_refs(self):
        self.refs[0].nxt = self.refs[1]
        
        if isinstance(self, Reference):
            length = len(self)
        elif isinstance(self, ReferenceTimeseries):
            length = super(ReferenceTimeseries, self).__len__()
        else:
            return TypeError

        for idx in range(1, length - 1):
            self.refs[idx].prev = self.refs[idx - 1]
            self.refs[idx].nxt = self.refs[idx + 1]

        self.refs[-1].prev = self.refs[-2]
            
    def __str__(self):
        out = ""
        for ref in self.refs:
            out += repr(ref) + "\n"

        return out

    def __len__(self):
        """
        Returns number of reference points given csv-files consists of
        """
        return len(self.refs)

    def __repr__(self):
        return "References as collection points"

class ReferenceTimeseries(References):
    """
    Extend the class References by implementing a series of values
    that hold reference values for a corresponding sequence of time.
    
    A reference for 5 timesteps could then look like:
    
    [Reference0, Reference1, Reference2, Reference3, Reference4]
    """

    # TODO: Change self.length to correspond to prediction horizon - must always have a reference!
    def __init__(self, ref_path, length, delta_t, time=0):
        self._ts_nr = 0 # Current timestep we're on (essentially curr_time // delta_t). Start at -1 so that the initial update leaves us ready at 0
        self.delta_t = delta_t
        self.length = length # Should correspond to Hp
        self.ref_series = [0] * self.length

        super().__init__(ref_path, time)
        
        self.update(update_time=False) # Do not update time during initialization!

    def update(self, update_time=True):
        """
        Updates the references-timeseries to be valid from current time
        and a _length_ number of timesteps into the future, and also
        increments the current time to account for updates being made
        """
        if update_time:
            self._ts_nr += 1
            self.curr_time += self.delta_t # Update _after_ because it's used to set during __init__
        
        self.ref_series[0] = self.curr_ref

        # t = self.curr_time
        for i in range(1, self.length): # Skip 0th element because it's set manually
            # t_in_future = t + (i * self.delta_t)
            self.ref_series[i] = self[i] # Add _ts_nr since the static self.refs is what's being accessed
    
    def __len__(self):
        """
        Returns how many steps into the future a series of reference
        values should reach
        """
        return self.length

    def __str__(self):
        out = ""
        for ref in self.ref_series:
            out += repr(ref) + "\n"

        return out

    def __repr__(self):
        return "References Timeseries"

    def __getitem__(self, key):
        """
        Returns the reference valid at the specified time in linear time(?)

        Note: The given key is multiplied, so that the API can stride with unit strides,
              and the size of the timestep will still be accounted for.
        
        Could probably be improved by better search method 
        """
        item = None
        key += self._ts_nr # Offset such that indexation is according to where we are in time
        key *= self.delta_t # Scale such that indexation matches magnitude of timesteps
        for ref in self.refs:
            # Avoids illegal comparison operation below
            if ref.nxt is None:
                item = ref
                break

            if key >= ref.time and key < ref.nxt.time:
                item = ref
                break

        return item
