from importlib.resources import files
import pandas as pd

def _load_csv(name: str) -> pd.DataFrame:
    with files('datasets').joinpath(name).open("rb") as f:
        return pd.read_csv(f)

WIND_DATA_10M = _load_csv("wind_dataset.csv")
WIND_DATA_20M = _load_csv("wind_dataset1.csv")
WIND_DATA_30M = _load_csv("wind_dataset2.csv")
WIND_DATA_40M = _load_csv("wind_dataset3.csv")
WIND_DATA_50M = _load_csv("wind_dataset4.csv")
TEXT_DATA = _load_csv("txtdata.csv")