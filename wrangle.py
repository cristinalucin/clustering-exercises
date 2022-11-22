import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import os
import numpy as np
import env

from env import user, password, host

def get_zillow_data():
    ''' Retrieve data from Zillow database within codeup, selecting specific features
    If data is not present in directory, this function writes a copy as a csv file. 
    '''
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        query = '''
            
        SELECT *
        FROM properties_2017
        LEFT JOIN propertylandusetype USING(propertylandusetypeid)
        JOIN predictions_2017 USING (parcelid)
        WHERE propertylandusedesc IN ("Single Family Residential",                       
                              "Inferred Single Family Residential")'''
        df = pd.read_sql(query, get_connection('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df
    
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
def remove_outliers(df,feature_list):
    ''' utilizes IQR to remove data which lies beyond 
    three standard deviations of the mean
    '''
    for feature in feature_list:
    
        #define interquartile range
        Q1= df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        #Set limits
        upper_limit = Q3 + 3 * IQR
        lower_limit = Q1 - 3 * IQR
        #remove outliers
        df = df[(df[feature] > lower_limit) & (df[feature] < upper_limit)]
    
    return df
    
def clean_zillow(df):
    ''' This function takes in zillow data, renames columns, replaces whitespace with nan values,
    and drops null values. This function returns a df.
    '''
    # Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df. 
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # Removes null values
    df = df.dropna()
    
    # Converting some columns from float to integers or objects
    df["fips"] = df["fips"].astype(int)
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    df["bedroomcnt"] = df["bedroomcnt"].astype(int)    
    df["lotsizesquarefeet"] = df["lotsizesquarefeet"].astype(int)
    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
    
    # Relabeling FIPS data
    df['county'] = df.fips.replace({6037:'Los Angeles',
                       6059:'Orange',
                       6111:'Ventura'})
    
    # Creating new column for home age using year_built, casting as integer
    df['home_age'] = 2017- df.yearbuilt
    df["home_age"] = df["home_age"].astype(int)
    
    # renaming columns
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'square_feet',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',
                              'lotsizesquarefeet' : 'lot_size',
                              'transactiondate' : 'transaction_date',
                              'parcelid' : 'parcel_id'}
                                )
    #remove outliers
    df = remove_outliers(df,['bedrooms','bathrooms','square_feet','tax_value'])
    
    return df

def train_validate_test_split(df, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 15% of the original dataset, validate is .1765*.85= 15% of the 
    original dataset, and train is 75% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.15, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.1765, 
                                       random_state=seed)
    return train, validate, test

