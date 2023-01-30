#!/usr/bin/env python3

# timer.py - fetched directly from: https://realpython.com/python-timer/#a-python-timer-class
#            and modified slightly

import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        self._total_time = 0

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._total_time += elapsed_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

    # TODO: Make into a decorator
    def lap(self, silent=False):
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        elapsed_time = time.perf_counter() - self._start_time
        self._total_time += elapsed_time
        self._start_time = None
        if not silent: print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        
        self._start_time = time.perf_counter()

    def total_time(self):
        print(f"Total time: {self._total_time:0.4f} seconds")