from exception_handling import exception_handling
from missing import missing_treatment
from dummies import get_dummies

def preprocess(df, num_vars, cat_vars):


    # exception case handling
    df = exception_handling(df)


    # missing treatment
    df = missing_treatment(df=df, num_vars=num_vars)


    # dummy creation
    df=get_dummies(df=df, cat_vars=cat_vars)

    return df


