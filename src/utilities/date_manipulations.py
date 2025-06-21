# importing modules
import pandas as pd

def last_sunday(
        df: pd.core.frame.DataFrame,
        date_column_name: str,
        sunday_date_column_name: str
) -> pd.core.frame.DataFrame:
    """
    This function takes in a dataframe and a column name and returns the same 
    dataframe with a column containing the date of it's last sunday
    
    Parameters
    ----------
     df: Pandas DataFrame
         A DataFrame containing dates
    date_column_name: String
        Column name of the DataFrame containing dates
    sunday_date_column_name: String
        Column name of the new last sunday dates
    
    Returns
    -------
        A DataFrame with last sunday date column
    """

    temp_df = df.copy()
    # convert date column to Timestamp
    temp_df[date_column_name] = pd.to_datetime(
        arg=temp_df[date_column_name],
        infer_datetime_format=True,
        errors="coerce"
    )
    # obtain last sunday date of each dates, if date is sunday return as it is
    temp_df[sunday_date_column_name] = (
            temp_df[date_column_name] -
            pd.TimedeltaIndex(
                data=temp_df[date_column_name].dt.dayofweek + 1,
                unit="D"
            )
    )

    return temp_df
