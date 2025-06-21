# importing modules
import numpy as np
import pandas as pd
import statistics as stat
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.regression.linear_model
from tqdm import tqdm

def vif(
    X: pd.core.frame.DataFrame,
    const: str = None
) -> pd.core.frame.DataFrame:
    """
    Calculates variance inflation factor (VIF), for all variables

    Variance Inflation Factor (VIF) quantifies the severity of multicollinearity in an ordinary least squares regression
    analysis.
    It provides an index that measures how much the variance (the square of the estimate's standard deviation) of an
    estimated regression coefficient is increased because of collinearity

    Parameters
    ----------
    X: Pandas DataFrame
        Design matrix with all explanatory variables, as for example used in regression
    const: string
        Column name containing values = 1 (Intercept Variable)
        If const is present, it should be a part of X
        Options are:
            1) None: Default, if no intercept present
            2) 'const': if intercept is added with sm.add_constant
            3) any other column name (as string) containing values = 1

    Returns
    -------
    Pandas DataFrame
        Variance inflation factor
    """

    # calculating VIF and converting result to a DataFrame
    vif = pd.DataFrame()
    vif["VIF"] = [
        variance_inflation_factor(X.values, i) \
        for i in tqdm(range(X.shape[1]))
    ]
    vif["Variable"] = X.columns
    
    if const is not None:
        vif = vif.loc[
            vif['Variable'] != const,
            ["Variable", "VIF"]
        ].reset_index(drop=True)
    else:
        vif = vif[["Variable", "VIF"]].reset_index(drop=True)
    
    vif = vif.sort_values(
        by=["VIF"],
        ascending=[False]
    ).reset_index(drop=True)

    return vif


def model_accuracies(
    df: pd.core.frame.DataFrame,
    dep_var: str,
    indep_vars: list,
    model: statsmodels.regression.linear_model.RegressionResultsWrapper,
    const: str = None,
) -> dict:
    """
    Creates a dictionary of model accuracies (MAE, MAPE, WMAPE)

    Parameters
    ----------
    df: Pandas DataFrame
        design matrix with all explanatory variables, as for example used in regression
    dep_var: string
        column name of the dependent variable
    indep_vars: list
        list of column names of independent variables
    model: statsmodels object. Type = RegressionResultsWrapper
        model equation stored in an object
    const: string
        Column name containing values = 1 (Intercept Variable)
        If const is present, it should be a part of X
        Options are:
            1) None: Default, if no intercept present
            2) 'const': if intercept is added with sm.add_constant
            3) any other column name (as string) containing values = 1

    Returns
    -------
    A dictionary
        A dictionary of accuracy metrices like MAE, MAPE & WMAPE
    """

    temp_df = df.copy()
    if const is not None:
        temp_df['predicted'] = model.predict(
            exog=sm.add_constant(temp_df[indep_vars])
        ).tolist()
    else:
        temp_df['predicted'] = model.predict(
            exog=temp_df[indep_vars]
        ).tolist()
        
    return dict(
        {
            'MAE': round(
                number=stat.mean(
                    np.absolute(temp_df[dep_var] - temp_df['predicted'])
                ),
                ndigits=2
            ),
            'MAPE': round(
                number=stat.mean(
                    (
                        np.absolute(
                            temp_df[dep_var] - temp_df['predicted']
                        ) / temp_df['predicted']
                    ) * 100
                ),
                ndigits=2
            ),
            'WMAPE': round(
                number=(
                    np.sum(
                        np.absolute(temp_df[dep_var] - temp_df['predicted'])
                    ) / np.sum(temp_df[dep_var])
                ) * 100,
                ndigits=2
            )
        }
    )

def model_results(
    model: statsmodels.regression.linear_model.RegressionResultsWrapper,
    train_data: pd.core.frame.DataFrame,
    indep_vars: list,
    dep_var: str,
    test_data: pd.core.frame.DataFrame = None,
    const: str = None,
    export_path: str = None
) -> pd.core.frame.DataFrame:
    """
    Creates a DataFrame of:
        1) model summary (beta coefficients, p-values, t-values, VIFs), and
        2) model accuracy (Adj. R-square, MAE, MAPE, WMAPE)

    Parameters
    ----------
    model: statsmodels object. Type = RegressionResultsWrapper
        model equation stored in an object
    train_data: Pandas DataFrame
        Training dataset
    test_data: Pandas DataFrame
        Test / Validation dataset
    indep_vars: list
        list of column names of independent variables
    dep_var: string
        column name of the dependent variable
    const: string
        column name containing values = 1 (Intercept Variable)
        If const is present, it should be a part of X
        Options are:
            1) None: Default, if no intercept present
            2) 'const': if intercept is added with sm.add_constant
            3) any other column name (as string) containing values = 1
    export_path: string
        exports model results in an excel file
        Default: None

    Returns
    -------
    Pandas DataFrame
        A DataFrame of:
            1) model summary (beta coefficients, p-values, t-values, VIFs), and
            2) model accuracy (Adj. R-square, MAE, MAPE, WMAPE)
    """

    try:
        temp = pd.read_excel(
            io=export_path,
            sheet_name="Sheet1",
            na_values=['#NA', '#N/A', '', ' ', 'na', 'NA']
        )
        iteration = temp['Iteration Number'].max() + 1
    except:
        iteration = 1

    result_table = pd.merge(
        left=pd.merge(
            left=pd.DataFrame(  # estimates of model parameters
                data={
                    'Iteration Number': iteration,
                    'Variable': model.params.index,
                    'Estimate': model.params
                }
            ).reset_index(drop=True),
            right=pd.DataFrame(  # p-values of model parameters
                data={
                    'Variable': model.pvalues.index,
                    'p-value': model.pvalues
                }
            ).reset_index(drop=True),
            how='left',
            left_on='Variable',
            right_on='Variable'
        ),
        right=pd.merge(
            left=pd.DataFrame(  # t-values of model parameters
                data={
                    'Variable': model.tvalues.index,
                    't-value': model.tvalues
                }
            ).reset_index(drop=True),
            right=vif(  # VIFs of model parameters
                X=train_data[indep_vars],
                const=const
            ),
            how='left',
            left_on='Variable',
            right_on='Variable'
        ),
        how='left',
        left_on='Variable',
        right_on='Variable'
    )
    result_table['Adj R-sq'] = model.rsquared_adj  # model Adjusted R-square
    # model error metrices
    result_table['Train MAE'] = model_accuracies(
        df=train_data,
        dep_var=dep_var,
        indep_vars=indep_vars,
        model=model,
        const=const
    )['MAE']
    result_table['Train MAPE'] = model_accuracies(
        df=train_data,
        dep_var=dep_var,
        indep_vars=indep_vars,
        model=model,
        const=const
    )['MAPE'] / 100
    result_table['Train WMAPE'] = model_accuracies(
        df=train_data,
        dep_var=dep_var,
        indep_vars=indep_vars,
        model=model,
        const=const
    )['WMAPE'] / 100
    if test_data != None:
        result_table['Test MAE'] = model_accuracies(
            df=test_data,
            dep_var=dep_var,
            indep_vars=indep_vars,
            model=model,
            const=const
        )['MAE']
        result_table['Test MAPE'] = model_accuracies(
            df=test_data,
            dep_var=dep_var,
            indep_vars=indep_vars,
            model=model,
            const=const
        )['MAPE'] / 100
        result_table['Test WMAPE'] = model_accuracies(
            df=test_data,
            dep_var=dep_var,
            indep_vars=indep_vars,
            model=model,
            const=const
        )['WMAPE'] / 100
    else:
        result_table['Test MAE'] = np.nan
        result_table['Test MAPE'] = np.nan
        result_table['Test WMAPE'] = np.nan
        
    print("Iteration Number: {}".format(iteration))

    # exporting model results
    if export_path is not None:  # if export is required
        if iteration > 1:  # if there is a file at 'export_path' location with previous iterations present
            result_table = pd.concat(
                [
                    temp,
                    result_table
                ],
                axis=0
            )
        else:  # if there is a file at 'export_path' location with no previous iterations present
            None
        with pd.ExcelWriter(
            path=export_path,
            mode='w',
            date_format='YYYY-MM-DD',
            datetime_format='DD-MMM-YYYY'
        ) as writer:
            result_table.to_excel(
                excel_writer=writer,
                index=False,
                sheet_name='Sheet1',
                engine='openpyxl'
            )
    else:  # if export is not required
        None

    return result_table
