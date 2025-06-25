import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utilities import missing_values

# Obtain full path for this file
project_name = r"predict_house_prices"
home_folder = os.path.abspath(__file__).split(project_name)[0] + project_name

sys.path.insert(0, home_folder + r"/src/utilities/")
import model_objects, missing_values

# Importing the data description JSON file
with open(home_folder + r"/data_description.json", "r") as file:
    data_description = json.load(file)

# Importing the train dataset
train = pd.read_csv(r'/Users/shreya/Library/CloudStorage/OneDrive-Personal/shared_projects/Predict House Prices/train.csv')
# Importing the test dataset
test = pd.read_csv(r'/Users/shreya/Library/CloudStorage/OneDrive-Personal/shared_projects/Predict House Prices/test.csv')

# changing the column names to lower case to match with data_description
train.columns = [col.lower() for col in train.columns.values.tolist()]
test.columns = [col.lower() for col in test.columns.values.tolist()]

# replace all empty values with np.nan. For example, '' is an empty value
train.replace("", np.nan, inplace=True)
test.replace("", np.nan, inplace=True)

# Treating missing values
for col in [
    "lotfrontage",
    "masvnrarea",
    "bsmtfinsf1",
    "bsmtfinsf2",
    "bsmtfullbath",
    "bsmthalfbath",
    "bsmtunfsf",
    "totalbsmtsf",
    "garagearea",
    "garagecars",
    "garageyrblt"
]:
    train[col]  = train[col].fillna(0)
    test[col]  = test[col].fillna(0)

# Check missing values
missing_values.check_missing_values(train)
missing_values.check_missing_values(test)

# Define the id and dependent variable column names
id = "id"
dep_var = "saleprice"
# List all numeric columns except id and dep_var
num_vars = [
    col
    for col, details in data_description.items()
    if details['dtype'] != 'object' and col not in [id, dep_var]
]
# List all categorical columns except id and dep_var
cat_vars = [
    col
    for col, details in data_description.items()
    if details['dtype'] == 'object' and col not in [id, dep_var]
]

# Convert categorical variables into dummy variables
train = pd.get_dummies(train, columns=cat_vars, drop_first=True, dtype=int)
test = pd.get_dummies(test, columns=cat_vars, drop_first=True, dtype=int)
# Ensure the train and test datasets have the same dummy variables
train, test = train.align(test, join='left', axis=1, fill_value=0)
test.drop(columns=[dep_var], inplace=True)
# Listing all dummy variables
dummy_vars = [
    col
    for col in train.columns.values.tolist()
    if col not in [id, dep_var] + num_vars + cat_vars
]

# Combine numeric and dummy variables
indep_vars = num_vars + dummy_vars


def model_res(dep, indep, train, test=None, const='const', export_path=None):
    # Define the dependent and independent variables for the model
    X = train[indep]
    y = train[dep]

    # Add a constant to the model (intercept)
    X = sm.add_constant(X)

    # Create the linear regression model
    model = sm.OLS(y, X).fit()

    # Print the model summary
    # print(model.summary())

    #Print Model Results
    return model_objects.model_results(
        model=model,
        train_data=train,
        test_data=None,
        dep_var=dep,
        indep_vars=indep,
        const='const',
        export_path=None
    )



v = 11
while v > 10:
    
    model_results = model_res(dep_var, indep_vars, train, test=None, const='const', export_path=None)
    
    v = model_results["VIF"].max()
    if v > 10:
        model_results.sort_values(by="VIF", ascending=False, inplace=True)
        indep_vars = model_results.iloc[1:model_results.shape[0]-1, :]["Variable"].tolist()
        print("'{}' eliminated. {} features remain".format(
            model_results.iloc[0, :]["Variable"],
            model_results.shape[0]-2
        ))
    else:
        pass

p = 1
while p > 0.05:

    model_results = model_res(dep_var, indep_vars, train, test=None, const='const', export_path=None)
    
    p = model_results["p-value"].max()
    if p > 0.05:
        model_results.sort_values(by="p-value", ascending=False, inplace=True)
        indep_vars = model_results.iloc[1:model_results.shape[0]-1, :]["Variable"].tolist()
        print("'{}' eliminated. {} features remain".format(
            model_results.iloc[0, :]["Variable"],
            model_results.shape[0]-2
        ))
    else:
        pass


display(model_results)

model_results["Base Variable"] = [i.split("_")[0] for i in model_results["Variable"]]


feature_group = {
    "Exterior Elements": ["exterior1st", "exterior2nd", "roofstyle", "roofmatl", "masvnrtype", "masvnrarea"],
    "Basement Features": ["bsmtfinsf2", "bsmtfintype2", "bsmtfintype1", "bsmtqual", "bsmtexposure", "bsmtunfsf", "bsmtfullbath"],
    "Neighborhood": ["neighborhood"],
    "Property Type": ["mssubclass", "bldgtype", "housestyle"],
    "Garage Features": ["garagequal", "garagetype"],
    "Outdoor Areas": ["openporchsf", "screenporch", "wooddecksf"],
    "Bathrooms": ["halfbath", "bsmtfullbath"],
    "Land Features": ["landcontour", "landslope", "lotconfig", "lotarea"],
    "Utilities": ["utilities"],
    "Heating System": ["heatingqc"],
    "Proximity to Conditions": ["condition1", "condition2"],
    "Fireplace": ["fireplacequ"],
    "Pool Features": ["poolqc", "poolarea"],
    "Foundation": ["foundation"],
    "Sale Conditions": ["salecondition"]
}

# map feature_group to "Base Variable" column in selected_features dataframe
reverse_feature_group_map = {}
for group, features in feature_group.items():
    for feature in features:
        reverse_feature_group_map[feature] = group

def feature_grouping(feature):
    if feature in reverse_feature_group_map:
        return reverse_feature_group_map[feature]
    else:
        return None

model_results['Group'] = model_results.apply(lambda row: feature_grouping(row["Base Variable"]), axis=1)




model_results.to_csv(r'/Users/shreya/Library/CloudStorage/OneDrive-Personal/shared_projects/Predict House Prices/p_val_elimination.csv', index=False)

model_results["Variable"].values


def format_y_tick(value, currency: str = False):
    if value >= 1_000_000_000:
        return currency + f'{value / 1_000_000_000:,.0f}B'
    elif value >= 1_000_000:
        return currency + f'{value / 1_000_000:,.0f}M'
    elif value >= 1_000:
        return currency + f'{value / 1_000:,.1f}K'
    else:
        return value

# plot a scatter plot between the two variables
import matplotlib.pyplot as plt
feature = r"bsmtfullbath"
plt.scatter(train[feature], train[dep_var])
# add a chart title
plt.title(f"{feature} vs {dep_var}")
plt.xlabel(feature)
plt.ylabel(dep_var)

# format the y-axis ticks as "#,##0 K", "#,##0 M", "#,##0 B"
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: format_y_tick(value, currency="$")))

plt.show()


format_y_tick(10000000, "$")


# test for linearity between the two variables
import seaborn as sns
sns.regplot(x=feature, y=dep_var, data=train.loc[train[feature]!=0,:])
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: format_y_tick(value, currency="$")))
plt.show()

# test for linearity between the two variables using F-test
from scipy.stats import f_oneway
for feature in num_vars:
    f_test_statistic, p_value = f_oneway(train.loc[train[feature]!=0,:][dep_var], train.loc[train[feature]==0,:][dep_var])
    print(f"Feature: {feature}, P-value: {p_value}")