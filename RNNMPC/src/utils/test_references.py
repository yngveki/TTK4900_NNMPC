#!/usr/bin/env python3

from references import ReferenceTimeseries
from pathlib import Path

if __name__ == "__main__":
    ref_path = Path(__file__).parent / "../../config/refs/testrefs.csv"
    length = 10
    delta_t = 1
    time = 0
    ref = ReferenceTimeseries(ref_path, length=length, delta_t=delta_t, time=time)
    print(ref)

    itr = 0
    while itr <= 100:
        ref.curr_time += delta_t
        ref.update()
        print(ref)

        itr += 1

    print("finished :)")