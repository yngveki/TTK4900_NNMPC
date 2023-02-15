#!/usr/bin/env python3
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Timeseries():

    def __init__(self, csv_path, delta_t=1):

        df = pd.read_csv(csv_path)
        assert len(df) > 0, "At least one entry must be given!"

        class Point():
            def __init__(self, time, choke, gl, dist):
                self.time = time
                self.choke = choke
                self.gl = gl
                self.dist = dist
        points = []
        for row in range(len(df)):
            data = df.iloc[row,:]
            if row + 1 >= len(df):
                timediff = None
            else:
                timediff = df.iloc[row+1][0] - data[0] # Time diff between entries

            points.append(Point(time=data[0], choke=data[1], gl=data[2], dist=timediff))

        # Interpolate between points
        self.begin = points[0].time
        self.end = points[-1].time
        self.length = (self.end - self.begin) // delta_t
        timeseries = [0] * self.length

        for point in points:
            num_points = point.dist // delta_t if point.dist is not None else 0
            for i in range(num_points):
                idx = (point.time // delta_t) + i
                timeseries[(point.time // delta_t) + i] = [point.choke, point.gl]

        self.timeseries = np.array(timeseries).T
        self.delta_t = delta_t

    def prepend(self, val, length):
        """Takes in [choke, gl] which will be prepended length times to self.timeseries"""
        
        assert isinstance(val, list), "Must be a list"
        assert len(val) == 2, "Timeseries class is only implemented for format: [choke, gl]"

        prep = np.array([val] * length)
        self.timeseries = np.insert(self.timeseries, 0, prep, axis=1)
        self.length = len(self)
        self.begin += length * self.delta_t
        self.end += length * self.delta_t
        return self.timeseries

    def plot(self):
        t = np.linspace(self.begin, self.end, self.length)
        fig, axes = plt.subplots(2, 1, sharex=True)
        fig.suptitle('Choke and gas lift rate over input over time')

        # Plotting choke input over time
        axes[0].set_title('Input: choke', fontsize=20)
        axes[0].set_ylabel('percent opening [%]', fontsize=15)
        axes[0].plot(t, self.timeseries[0], '-', label='choke', color='tab:red')
        axes[0].legend(loc='best', prop={'size': 15})

        # Plotting gas lift rate input over time
        axes[1].set_title('Input: gas lift rate', fontsize=20)
        axes[1].set_xlabel('time [s]', fontsize=15)
        axes[1].set_ylabel('gas rate [m^3/h]', fontsize=15)
        axes[1].plot(t, self.timeseries[1], label='gas lift rate', color='tab:blue')
        axes[1].legend(loc='best', prop={'size': 15})

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        plt.show(block=False)
        plt.pause(30)
        plt.close()

    def __getitem__(self, key):
        return self.timeseries[key]

    def __len__(self):
        assert len(self.timeseries[0]) == len(self.timeseries[1]), "Timeseries must be equally long for both inputs"
        return len(self.timeseries[0])