import numpy as np
import math
from datetime import datetime


def score_fn(anomaly_ts, anomaly_ranges):
    # Initialization
    Atp = 1
    Afn = -1
    Afp = -0.11
    fd = 0
    scores = []
    results = []
    undetected_windows = set(anomaly_ranges)
    for ts in anomaly_ts:
        for start, end in anomaly_ranges:
            if start <= ts <= end:
                if (start, end) not in scores:
                    # First anomaly in this range, assign +1
                    scores.append((start, end))
                    undetected_windows.discard((start, end))
                    results.append(Atp)
                    break
                else:
                    # Subsequent anomaly in this range, assign 0
                    results.append(0)
                    break
        else:
            # Anomaly outside of any range, calculate the distance to the nearest window to the left
            left_windows = [(end, (end-start).total_seconds()) for start, end in anomaly_ranges if end < ts]
            if left_windows:
                nearest_left_window, window_width = max(left_windows, key=lambda x: x[0])
                y = (ts - nearest_left_window).total_seconds() / window_width
                sigma = (2 / (1 + math.exp(5 * y))) - 1
                results.append(sigma * -Afp) 
            else:
                results.append(Afp)

    fd = len(undetected_windows)
    results.append(Afn*fd)
    return results



# anomaly_ranges = [
#     ["2015-03-08 21:02:53.000000","2015-03-10 17:12:53.000000"],
#     ["2015-03-19 01:02:53.000000","2015-03-20 21:12:53.000000"],
#     ["2015-03-25 21:02:53.000000","2015-03-27 17:12:53.000000"]
# ]

# anomaly_ts = ["2015-03-07 21:02:53.000000", "2015-03-08 22:02:53.000000","2015-03-09 22:02:53.000000",
#               "2015-03-09 23:02:53.000000","2015-03-24 21:02:53.000000"]
# # Convert the generator to a list to get all the scores
# scores =score_fn(anomaly_ts, anomaly_ranges)
# print(f'Score: {np.sum(scores)}')
# print(scores)