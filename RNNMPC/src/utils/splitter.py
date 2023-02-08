#!/usr/bin/env python3

# Implements a class to take in intervals of values and return,
# iteratively, configurations of the values within, given some 
# resolution

class Interval():
    """Defines an interval, which defines a set of values, given some resolution"""

    def __init__(self, interval, resolution):
        """
        Initializes an interval

        Args:
            :param interval: a 2-long list, containing start and end values for interval
            :param resolution: the amount of values the interval should be split into
        """

        self.begin = interval[0]
        self.end = interval[-1]
        diff = self.end - self.begin
        if resolution > diff:
            resolution = diff + 1 # e.g. mu=[8,10], res=5, doesn't make sense; need only test 8, 9 and 10, then
        self.incr = diff / (resolution - 1) # - 1, since we start _at_ begin, and want to end _at_ end

        self.curr = self.begin
        self._nxt = self.curr + self.incr

        self.looped = False

    @property
    def nxt(self):
        return self._nxt

    @nxt.setter
    def nxt(self, new_val):
        if new_val <= self.end:
            self._nxt = new_val
        else:
            self._nxt = None
        # assert self.nxt + self.incr <= self.end, "Next value cannot supercede given end-value!\n"
        # self._nxt += increment

    def iterate(self):
        if self.nxt is None:
            self.looped = True
            return None
        
        self.curr = self.nxt
        self.nxt = self.nxt + self.incr
        return self.nxt
        
class Splitter():
    """Iteratively extracts configurations of values within given intervals"""

    def __init__(self, resolution, *args, **kwargs):
        self.curr_iterating = None
        self.nxt_iterating = None

        self.intervals = {}
        self.keys = []
        for key, value in kwargs:
            if isinstance(value, list):
                self.intervals[key] = Interval(value, resolution)
                self.keys.append(key)

        if any(self.keys): # List is not empty
            self.itr = iter(self.keys)
            self.curr_iterating = next(self.itr) # Current key that we're iterating over

    # @property
    # def nxt_iterating(self):
    #     return self._nxt_iterating

    # @nxt_iterating.setter
    # def nxt_iterating(self, new_val):
    #     try:
    #         nxt = next(self.itr)
    #     except StopIteration:
    #         nxt = None
    #     self._nxt_iterating = nxt

    def iterate(self):
        try:
            nxt = next(self.itr)
        except StopIteration:
            nxt = None
        self.nxt_iterating = nxt

    # TODO: Now you've made a framework for iterating over intervals.
    # TODO: Now make the API which let's the user simply say "I want the next value"

    def get_next_config(self):
        ...


# ----- For testing ----- #
if __name__ == "__main__":
    res = 5
    interval = Interval([0, 10], res)
    for i in range(res):
        print(interval.curr)
        print(interval.nxt)
        interval.iterate()

    if 'key':
        print('key')
    splitter = Splitter(res, mu=[8,10], my=[8,12])
