import numpy as np
import pandas as pd

from load_data import load_data
from exception_handling import exception_handling
from missing import missing_treatment
from dummies import get_dummies

train_df = load_data("train")
data_description = load_data("data_dictionary")


# Define the id and dependent variable column names
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


# exception case handling
train_df = exception_handling(train_df)


# missing treatment
train_df = missing_treatment(df=train_df, num_vars=num_vars)


# dummy creation
train_df=get_dummies(df=train_df, cat_vars=cat_vars)

display(train_df.head())

