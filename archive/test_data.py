import unittest
import pandas as pd

from archive.data import Data
from archive.stat import Stat
from happysimulator.utils.instant import Instant


class TestData(unittest.TestCase):
    def test_add_stat(self):
        data = Data()
        t1 = Instant.from_seconds(1)
        data.add_stat(10, t1)
        self.assertEqual(len(data.df), 1)
        self.assertEqual(data.df.loc[0, data.STAT], 10)
        self.assertEqual(data.df.loc[0, data.TIME_NANOS], t1.nanoseconds)

    def test_get_stats(self):
        data = Data()
        t1 = Instant.from_seconds(1)
        t2 = Instant.from_seconds(2)
        t3 = Instant.from_seconds(3)
        data.add_stat(10, t1)
        data.add_stat(20, t2)
        data.add_stat(100000, t3)

        avg_stat = data.get_stats(t1, t3, Stat.AVG) # inclusive of t1, exclusive of t3
        self.assertEqual(avg_stat, 15)

        no_stat = data.get_stats(Instant.from_seconds(4), Instant.from_seconds(7), Stat.AVG)
        self.assertTrue(pd.isna(no_stat))

    def test_flush_stats(self):
        data = Data()
        t1 = Instant.from_seconds(1)
        t2 = Instant.from_seconds(2)
        data.add_stat(10, t1)
        data.add_stat(20, t2)

        data.flush_stats(t2)  # Flush stats before t2, should remove the stat at t1
        self.assertEqual(len(data.df), 1)
        self.assertEqual(data.df.loc[0, data.STAT], 20)
        self.assertEqual(data.df.loc[0, data.TIME_NANOS], t2.nanoseconds)


if __name__ == '__main__':
    unittest.main()
