
import pandas as pd


def get_dummies(df, cat_vars):
    null_cats = df[cat_vars].isnull().sum()
    null_cats=null_cats[null_cats > 0].index.tolist()
    
    non_null_cats = [i for i in cat_vars if i not in null_cats]

   # Convert categorical variables into dummy variables
    df = pd.get_dummies(
        data=df,
        columns=non_null_cats,
        drop_first=True, 
        dtype=int
    )

    df = pd.get_dummies(
        data=df,
        columns=null_cats, 
        dummy_na=False,
        drop_first=False, 
        dtype=int
    )

    return df




if __name__ == "__main__":
    from load_data import load_data

    train_df = load_data("train")
    data_description = load_data("data_dictionary")

    id_col = "id"
    dep_var = "saleprice"
    column_dtypes = data_description.drop_duplicates("column_name").set_index("column_name")["dtype"]

    cat_vars = [
        col
        for col, dtype in column_dtypes.items()
        if dtype == "object" and col not in [id_col, dep_var]
    ]

    temp = get_dummies(df=train_df, cat_vars=cat_vars)
    print(temp.shape)