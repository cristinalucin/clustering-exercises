import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import os
import numpy as np
import env

from env import user, password, host

##--------------------Get Data------------------##

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def acquire_zillow():
    ''' Retrieve data from Zillow database within codeup, selecting specific features
    If data is not present in directory, this function writes a copy as a csv file. 
    '''
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        query = '''SELECT *
        FROM properties_2017
        LEFT JOIN predictions_2017 USING (parcelid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN unique_properties USING (parcelid)
        WHERE transactiondate LIKE '2017%%'
        AND latitude is not NULL
        AND longitude is not NULL
        AND (propertylandusetypeid = 261 OR propertylandusetypeid = 279);'''
        df = pd.read_sql(query, get_connection('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df
    
##-------------------Summarize-------------------##

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    '''
    print('SUMMARY REPORT')
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3))
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts(), '\n')
        else:
            print(df[col].value_counts(bins=10, sort=False), '\n')
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('=====================================================')
    
##-------------------Null Values-----------------##
    
def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing/ rows *100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing' : percent_missing})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)

def nulls_by_rows(df):
    '''This function takes in a df and calculates the number of rows with null values
    and the percentage of rows with null values, returns a df with these values in columns'''
    num_missing = df.isnull().sum(axis=1)
    percent_miss = num_missing / df.shape[1]*100
    rows = df.shape[0]
    percent_missing = num_missing/ rows *100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing' : percent_missing})
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)

def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    '''This function drop rows or columns based on the percent of values that are missing,
    dropping columns before rows'''
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    return df

