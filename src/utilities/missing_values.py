# importing modules
import pandas as pd

def check_missing_values(
        df: pd.core.frame.DataFrame,
        columns: list = 'all'
) -> pd.core.frame.DataFrame:
    """
    Calculates count and percentage of missing values per columns in a pandas DataFrame

    Parameters
    ----------
    df: Pandas DataFrame
        A pandas DataFrame to check for missing values
    columns: list, default 'all'
        Columns to check for missing values
        If default, then all columns will be checked

    Returns
    -------
    A DataFrame
        A DataFrame showing count and percentage of missing values per selected column
    """

    if columns == 'all':
        columns = df.columns.values.tolist()
    # calculating percentage of values missing in the DataFrame
    missing_values = pd.DataFrame(
        {
            'Count Missing': len(df.index) - df[columns].count(),
            'Percentage Missing': ((len(df.index) - df[columns].count()) / len(df.index)) * 100
        }
    )
    # sorting percentage missing in descending order
    missing_values.sort_values(
        by="Percentage Missing",
        ascending=False,
        axis=0,
        inplace=True
    )

    return missing_values
