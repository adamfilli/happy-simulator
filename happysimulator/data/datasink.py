import pandas as pd
from typing import List, Union, Optional

from happysimulator.utils.instant import Instant
from happysimulator.utils.filename import sanitize_filename


class DataSink:
    def __init__(self, name: str, stat_names: Union[str, List[str]]):
        if isinstance(stat_names, str):
            stat_names = [stat_names]  # Convert to list for uniform processing
        self.stat_names = stat_names  # Store as list
        self.df = pd.DataFrame(columns=['TIME_SECONDS'] + self.stat_names)
        self.name = name

    def add_stat(self, time: Instant, stats: Union[float, dict]):
        if isinstance(stats, dict):
            # Ensure stats dict only contains keys that are in self.stat_names
            filtered_stats = {k: v for k, v in stats.items() if k in self.stat_names}
            new_row = {'TIME_SECONDS': time.to_seconds(), **filtered_stats}
        else:
            if len(self.stat_names) != 1:
                raise ValueError("A single stat value provided but multiple stat names are defined.")
            new_row = {'TIME_SECONDS': time.to_seconds(), self.stat_names[0]: stats}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def generate_csv(self) -> str:
        return f"{self.name}\n" + self.df.to_csv(index=False)

    def save_csv(self, directory: Optional[str] = None, filepath: Optional[str] = None):
        if directory is not None and filepath is not None:
            raise ValueError("Set one of either directory or filepath but not both")

        if directory is not None:
            self.df.to_csv(f"{directory}/{sanitize_filename(self.name)}.csv", index=False)
            return

        if filepath is not None:
            self.df.to_csv(filepath, index=False)
            return