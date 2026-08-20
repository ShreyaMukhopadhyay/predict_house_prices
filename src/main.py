import numpy as np
import pandas as pd

from load_data import load_data
from data_prep import preprocess




"""
STEP 1: LOAD DATSET
"""
train_df = load_data("train")
data_description = load_data("data_dictionary")


"""
STEP 2: Define the id and dependent variable column names
"""
id_col = "id"
dep_var = "saleprice"

# One dtype per column
column_dtypes = data_description.drop_duplicates("column_name").set_index("column_name")["dtype"]

# List all numeric columns except id and dep_var
num_vars = [
    col
    for col, dtype in column_dtypes.items()
    if dtype != "object" and col not in [id_col, dep_var]
]
# List all categorical columns except id and dep_var
cat_vars = [
    col
    for col, dtype in column_dtypes.items()
    if dtype == "object" and col not in [id_col, dep_var]
]


"""
STEP 3: PROCESS DATASET
"""
modeling_ad = preprocess(train_df, num_vars, cat_vars)




display(modeling_ad.head())

