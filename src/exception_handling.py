def exception_handling(df):
    df = df[~df['electrical'].isna()]

    return df