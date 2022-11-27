import pandas as pd
import numpy as np

from env import get_db_url

def get_zillow_data():
    '''This function retrieves data from the Codeup mySQL database (zillow)'''
    sql_query = '''
    SELECT * FROM properties_2017
    JOIN predictions_2017 USING (parcelid)
    WHERE transactiondate < '2018'
    AND propertylandusetypeid = 261;
    '''
    
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    df = df.drop(columns='id')
    
    return df

