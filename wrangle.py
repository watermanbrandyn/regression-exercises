# Wrangle

# os needed to do local inspection of cache, to see if data exists locally
import os
# env.py contains our credentials to access the SQL server we are pulling the data from
from env import host, user, password
# Pandas is needed to perform SQL interaction
import pandas as pd
import numpy as np

# Splitting function
from sklearn.model_selection import train_test_split

# Acquire Functions

def get_db_url(db_name, username=user, hostname=host, password=password):
    '''
    This function requires a database name (db_name) and uses the imported username,
    hostname, and password from an env file. 
    A url string is returned using the format required to connect to a SQL server.
    '''
    url = f'mysql+pymysql://{username}:{password}@{host}/{db_name}'
    return url

def get_zillow_data(use_cache= True):
    '''
    
    '''
    # Checking to see if data already exists in local csv file
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    # If data is not local we will acquire it from SQL server
    print('Acquiring data from SQL db')
    # Query to refine what data we want to grab 
    # bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips from properties_2017 
    # where propertylandusetypeid == 261 (Single Family Residential)
    query = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261 OR propertylandusetypeid = 279 
    '''
    # Command line interaction with SQL server and assignment to dataframe (df)
    df = pd.read_sql(query, get_db_url('zillow'))
    # Creation of csv file
    df.to_csv('zillow.csv', index=False)
    # Returns the dataframe
    return df


def prep_zillow(df):
    '''
    
    '''
    # Dropping duplicates in df
    df.drop_duplicates(inplace=True)
    # Dropping null values
    df = df.dropna()
    # Converting some of our columns to more appropriate data types
        # bedroomcnt, yearbuilt, calculatedfinishedsquarefeet to int
        # fips to str
    to_int = ['bedroomcnt', 'yearbuilt', 'calculatedfinishedsquarefeet']
    # Looping through our specified lists to change the data type
    for col in to_int:
        df[col] = df[col].astype(int)
    # Ensuring that the '0' remains for the fips column after converting to str type
        # Convert to an int first to remove the trailing decimal
    df.fips = df.fips.astype(int)
    df.fips = df.fips.astype(str)
    df.fips = '0' + df.fips
    # Distinguishing our categorical columns
    cat_cols = [col for col in df.columns if df[col].dtype == 'O']
    # Encoding our categorical columns
    for col in cat_cols:
        # Create the dummy df
        dummy_df = pd.get_dummies(df[col],
                            prefix = df[col].name,
                            drop_first = True,
                            dummy_na = False)
        # Add the dummy vars to the df
        df = pd.concat([df, dummy_df], axis=1)
    # Split our df in to train, validate, and test splits (3 df)
    train, validate, test = zillow_split(df)
    return train, validate, test


def zillow_split(df):
    '''
    This function takes in a dataframe and returns train, validate, test splits. (dataframes)
    An initial 20% of data is split to place as 'test'
    A second split is performed (on the remaining 80%) between train and validate (70/30)
    '''
    # First split with 20% going to test
    train_validate, test = train_test_split(df, train_size = .8, 
                                                 random_state = 123)
    # Second split with 70% of remainder going to train, 30% to validate
    train, validate = train_test_split(train_validate, train_size = .7,
                                                random_state=123)
    # Return train, validate, test (56%, 24%, 20% splits of original df)
    return train, validate, test


def wrangle_zillow():
    # Acquire the data
    df = get_zillow_data()
    # Prep the data (returns train, validate, test)
    return prep_zillow(df)