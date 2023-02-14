#!/usr/bin/env python3

from references import ReferenceTimeseries
from timeseries import Timeseries
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    ref_path = Path(__file__).parent / "../../generate_data/steps/steps100k.csv"
    timeseries = Timeseries(ref_path, delta_t=10)
    # timeseries.strip()
    # timeseries.prepend(length=5, val=[3.14,6.28])
    # timeseries.transpose()
    # length = 10
    # delta_t = 1
    # time = 0
    # ref = ReferenceTimeseries(ref_path, length=length, delta_t=delta_t, time=time)
    # print(ref)

    # itr = 0
    # while itr <= 100:
    #     ref.curr_time += delta_t
    #     ref.update()
    #     print(ref)

    #     itr += 1
    # timeseries.plot()

    val = [2,10]
    length = 3
    mock = np.zeros((2,5))
    prep = np.array([val] * length)
    timeseries = np.insert(mock, 0, prep, axis=1)
    print(timeseries)
    print("finished :)")