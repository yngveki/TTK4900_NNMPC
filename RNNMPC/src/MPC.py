#!/usr/bin/env python3

class RNNMPC:

    def __init__(self):
        return NotImplementedError

    def warm_start(self):
        return NotImplementedError

    def update_OCP(self):
        return NotImplementedError

    def solve_OCP(self):
        return NotImplementedError

    def iterate_system(self):
        return NotImplementedError