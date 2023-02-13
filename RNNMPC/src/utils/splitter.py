#!/usr/bin/env python3

# Implements a class to take in intervals of values and return,
# iteratively, configurations of the values within, given some 
# resolution

import warnings

class Node():

    def __init__(self, data, prev=None, nxt=None):
        self.data = data
        self.prev = prev
        self.nxt = nxt

class Interval():
    """Defines an interval, which defines a set of values, given some resolution"""

    def __init__(self, key, interval, resolution):
        """
        Initializes an interval

        Args:
            :param key: a string to associate the interval to some specific variable
            :param interval: a 2-long list, containing start and end values for interval
            :param resolution: the amount of values the interval should be split into
        """

        self.key = key
        self.begin = interval[0]
        self.end = interval[-1]
        diff = self.end - self.begin
        if resolution > diff:
            resolution = diff + 1 # e.g. mu=[8,10], res=5, doesn't make sense; need only test 8, 9 and 10, then
        self.incr = diff / (resolution - 1) # - 1, since we start _at_ begin, and want to end _at_ end

        self.values = []
        curr = self.begin
        for i in range(resolution):
            self.values.append(curr)
            curr += self.incr
        # self.curr = self.begin
        # self._nxt = self.curr + self.incr

        # self.looped = False

    # @property
    # def nxt(self):
    #     return self._nxt

    # @nxt.setter
    # def nxt(self, new_val):
    #     if new_val <= self.end:
    #         self._nxt = new_val
    #     else:
    #         self._nxt = None
    #     # assert self.nxt + self.incr <= self.end, "Next value cannot supercede given end-value!\n"
    #     # self._nxt += increment

    # def iterate(self):
    #     if self.nxt is None:
    #         self.looped = True
    #         return None
        
    #     self.curr = self.nxt
    #     self.nxt = self.nxt + self.incr
    #     return self.nxt

    # def reset(self):
    #     self.curr = self.begin
    #     self.nxt = self.curr + self.incr
        
class Splitter():
    """Iteratively extracts configurations of values within given intervals"""

    def __init__(self, resolution, **kwargs):
        warnings.warn("""Splitter is not fully implemented, and spits out only 
                         first possible configuration of given intervals""")
        # self.curr_iterating = None
        # self.nxt_iterating = None
        # self.configs = None
        # self.iter_limit = 0

        self.intervals = {}
        self.config = []
        # self.keys = []
        for key, value in kwargs.items():
            if isinstance(value, list):
                # self.intervals[key] = Interval(value, resolution)
                interval = Interval(key, value, resolution)
                self.intervals[key] = Node(interval)
                # self.keys.append(key)

        # if any(self.keys): # List is not empty
        #     self.itr = iter(self.keys)
        #     self.curr_iterating = next(self.itr) # Current key that we're iterating over

        #     self.configs = self._build_configs()
        #     self.iter_limit = len(self.keys)

        self.config = [interval.begin for interval in self.intervals]

        return self.config

    # def _build_configs(self):
    #     configs = []

    #     def recurse(current, depth):
    #         if depth != 0:
    #             for interval
    #     for key, interval in self.intervals:
    #         ...

    # def collect_folders(start, depth=-1):
    #     """ negative depths means unlimited recursion """
    #     folder_ids = []

    #     # recursive function that collects all the ids in `acc`
    #     def recurse(current, depth):
    #         folder_ids.append(current.id)
    #             if depth != 0:
    #                 for folder in getChildFolders(current.id):
    #                     # recursive call for each subfolder
    #                     recurse(folder, depth-1)

    #     recurse(start, depth) # starts the recursion
    #     return folder_ids

    # def _iterate(self):
    #     try:
    #         nxt = next(self.itr)
    #     except StopIteration:
    #         nxt = None
    #     self.nxt_iterating = nxt

    # API for use, such that user may simply say: "I want the next value"
    # def get_next_config(self):
    #     #! Exponential runtime wrt resolution and amount of intervals
    #     configs = []
    #     for interval in self.intervals
    #     config = {}
    #     for idx, key in enumerate(self.keys):
    #         config[key] = self.intervals[key].curr
    #         # TODO: Need to 
    #         self.intervals[key].iterate()

    #     return config

    def __iter__(self):
        self.curr_idx = 0
        return self

    def __next__(self):
        key = self.keys[self.curr_idx]
        curr_interval = self.intervals[self.keys[key]]

        if self.curr_idx > self.iter_limit:
            raise StopIteration

        self.curr_idx += 1
        return curr_interval 

# ----- For testing ----- #
if __name__ == "__main__":
    res = 5
    interval = Interval([0, 10], res)
    for i in range(res):
        print(interval.curr)
        print(interval.nxt)
        interval.iterate()

    splitter = Splitter(res, mu=[8,10], my=[8,12])

#? Some attempt at solving above problem (iterating over configs) recursively. Think I'll trash
# def rec(self, iter):
#     try:
#         res = rec(next(iter))
#     except StopIteration:
#         for val in interval:

#         return res