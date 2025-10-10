import os
import sys
import json
import pandas as pd
import sqlite3
# Define the path to the SharePoint directory
sharepoint_path = os.getenv('HOME') + r"/Github"

# Importing the data description JSON file
with open(sharepoint_path + r"/predict_house_prices/data_description.json", "r") as file:
    data_description = json.load(file)

# SQLite connection
def import_table(table, database):
    # Connect to the SQLite database
    conn = sqlite3.connect(database)
    # Read data into a DataFrame
    df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
    return df

# Importing the train dataset from SQL database
train = import_table(
    r"train",
    r"/Users/wrngnfreeman/Library/CloudStorage/OneDrive-Personal/shared_projects/sql_databases/house_prices.db"
)
# Function to split DataFrame based on data_description.json
def split_dataframe(df, data_description):
    splits = {}
    for table_name, fields in data_description.items():
        if isinstance(fields, list):
            # Create a subset of the DataFrame using the specified fields
            splits[table_name] = df[fields]
    return splits

# Split the train DataFrame based on the data description
split_dfs = split_dataframe(train, data_description)

# Function to create and populate a new SQLite database
def create_and_populate_db(splits, db_path):
    # Connect to the new SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect(db_path)

    for table_name, df in splits.items():
        # Write each DataFrame to a separate table in the new SQLite database
        df.to_sql(table_name, conn, if_exists='replace', index=False)

    conn.close()

# Define the path for the new SQLite database
new_db_path = r"/Users/wrngnfreeman/Library/CloudStorage/OneDrive-Personal/shared_projects/sql_databases/new_house_prices.db"

# Create and populate the new SQLite database
create_and_populate_db(split_dfs, new_db_path)
