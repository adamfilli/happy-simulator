import sys

import pandas as pd

from .stat import Stat
from ..happysimulator.utils.instant import Instant


class Data:
    STAT = 'stat'
    TIME_NANOS = 'time_nanos'
    TIME_PERIOD = 'time_period'

    def __init__(self):
        self.df = pd.DataFrame(columns=[self.TIME_NANOS, self.STAT])

    def add_stat(self, stat: int, time: Instant) -> None:
        new_row = {self.TIME_NANOS: time.nanoseconds, self.STAT: stat}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def get_stats(self, begin: Instant, end: Instant, aggregator: Stat) -> float:
        filtered_df = self.df[(self.df[self.TIME_NANOS] >= begin.nanoseconds) & (self.df[self.TIME_NANOS] < end.nanoseconds)]

        if not filtered_df.empty:
            aggregated_value = aggregator.aggregate(filtered_df[self.STAT])
            return aggregated_value
        else:
            return float('nan')

    def flush_stats(self, cutoff: Instant = None):
        if cutoff is None:
            cutoff = Instant.Epoch

        self.df = self.df[self.df[self.TIME_NANOS] >= cutoff.nanoseconds].reset_index(drop=True)

    def print_csv(self) -> None:
        self.df.to_csv(sys.stdout, index=False)