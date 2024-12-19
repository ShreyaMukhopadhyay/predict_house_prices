import pandas as pd
from sqlalchemy import create_engine

# MySQL connection
user = "root"
password = "Pgl-9U5-a52"
host = "192.168.2.41"
database = "house_prices"
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database}')

## Importing datasets to python
train = pd.read_sql(
    sql="SELECT * FROM train;",
    con=engine
)
test = pd.read_sql(
    sql="SELECT * FROM test;",
    con=engine
)

id = "id"
dep_var = "saleprice"

num_vars = []
cat_vars = []