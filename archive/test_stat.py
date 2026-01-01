import unittest
import pandas as pd

from archive.stat import Stat


class TestStat(unittest.TestCase):

    def setUp(self):
        self.data = pd.Series([1, 2, 3, 4, 5, pd.NA]) # pd.NA is ignored
        self.empty_data = pd.Series([])
        self.nan_data = pd.Series([pd.NA, pd.NA])

    def test_count(self):
        self.assertEqual(Stat.COUNT.aggregate(self.data), 5)
        self.assertEqual(Stat.COUNT.aggregate(self.empty_data), 0)
        self.assertEqual(Stat.COUNT.aggregate(self.nan_data), 0)

    def test_sum(self):
        self.assertEqual(Stat.SUM.aggregate(self.data), 15)
        self.assertEqual(Stat.SUM.aggregate(self.empty_data), 0)
        self.assertEqual(Stat.SUM.aggregate(self.nan_data), 0)

    def test_avg(self):
        self.assertEqual(Stat.AVG.aggregate(self.data), 3)
        self.assertTrue(pd.isna(Stat.AVG.aggregate(self.empty_data)))
        self.assertTrue(pd.isna(Stat.AVG.aggregate(self.nan_data)))

if __name__ == '__main__':
    unittest.main()