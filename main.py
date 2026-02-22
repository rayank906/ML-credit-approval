import pandas as pd

application_df = pd.read_csv("application_record.csv", header=0)
credit_df = pd.read_csv("credit_record.csv", header=0)

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

dropped_df.to_csv("joined_no_status.csv", index=False)