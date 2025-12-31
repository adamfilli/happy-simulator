from enum import Enum
import pandas as pd


class Stat(Enum):
    COUNT = 1
    SUM = 2
    AVG = 3
    P0 = 4
    P50 = 5
    P75 = 6
    P90 = 7
    P99 = 8
    P999 = 9
    P9999 = 10
    P100 = 11

    def __str__(self):
        return self.name  # Returns the name of the enum member, e.g., "AVG" for Stat.AVG

    @property
    def string_name(self) -> str:
        return self.name

    def aggregate(self, data: pd.Series) -> float:
        if self == Stat.COUNT:
            return data.count()
        elif self == Stat.SUM:
            return data.sum()
        elif self == Stat.AVG:
            return data.mean()
        elif self == Stat.P0:
            return data.quantile(0)  # Min
        elif self == Stat.P50:
            return data.quantile(0.5)  # Median
        elif self == Stat.P75:
            return data.quantile(0.75)
        elif self == Stat.P90:
            return data.quantile(0.9)
        elif self == Stat.P99:
            return data.quantile(0.99)
        elif self == Stat.P999:
            return data.quantile(0.999)
        elif self == Stat.P9999:
            return data.quantile(0.9999)
        elif self == Stat.P100:
            return data.quantile(1)  # Max
        else:
            raise NotImplementedError("Stat type not implemented")
