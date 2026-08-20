
def missing_treatment(df,num_vars):

    # Neighborhood-level median lotfrontage, computed from train only to avoid leakage
    lotfrontage_medians = df.groupby("neighborhood")["lotfrontage"].median()

    df["lotfrontage"] = df["lotfrontage"].fillna(
        df["neighborhood"].map(lotfrontage_medians)
    )

    df["garageyrblt"] = df["garageyrblt"].fillna(
        df["yearbuilt"]
    )
    
    for col in num_vars:
        if col != "lotfrontage":
            df[col] = df[col].fillna(0)
    return df
