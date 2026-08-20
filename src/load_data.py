import pandas as pd
from sqlalchemy import create_engine

from utilities.load_env import PG_HOST, PG_PORT, PG_USER, PG_PASSWORD

TARGET_DB = "house_price_prediction"


def get_target_engine():
    return create_engine(
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{TARGET_DB}"
    )


def load_data(dataset: str = "train") -> pd.core.frame.DataFrame:
    """
    Loads the 'train' or 'test' table from the house_price_prediction PostgreSQL database

    Parameters
    ----------
    dataset: str, default 'train'
        Which table to load. Must be either 'train' or 'test'

    Returns
    -------
    A DataFrame
        The requested table as a pandas DataFrame
    """

    if dataset not in ("train", "test"):
        raise ValueError("dataset must be either 'train' or 'test'")

    engine = get_target_engine()
    try:
        df = pd.read_sql_table(dataset, engine)
    finally:
        engine.dispose()

    return df


if __name__ == "__main__":
    train = load_data("train")
    print(f"Loaded 'train' table: {len(train)} rows")

    test = load_data("test")
    print(f"Loaded 'test' table: {len(test)} rows")
