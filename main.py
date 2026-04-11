from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent
_DATA = _REPO / "data"


def _csv(name: str) -> Path:
    p = _DATA / name
    return p if p.exists() else _REPO / name


application_df = pd.read_csv(_csv("application_record.csv"), header=0)
credit_df = pd.read_csv(_csv("credit_record.csv"), header=0)

joined_df = application_df.merge(credit_df, on="ID", how="inner")

def status_to_default(s):
    if s == "X":
        return 0
    if str(s).isdigit():
        return 1 if int(s) >= 3 else 0  
    if s == "C":
        return 0
    return 0

joined_df = joined_df.assign(STATUS_NUMERIC=joined_df["STATUS"].apply(status_to_default))

dropped_df = joined_df.drop(columns=["STATUS"])

_DATA.mkdir(parents=True, exist_ok=True)
dropped_df.to_csv(_DATA / "joined_no_status.csv", index=False)