# local_host
from env import user, password, host
import os

# python data science library's
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,classification_report,mean_squared_error, r2_score, explained_variance_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import f_regression, SelectKBest, RFE


# visualizations
from pydataset import data
import matplotlib.pyplot as plt
import seaborn as sns

def get_db_url(db_name, username=user, hostname=host, password=password):
    '''
    This function requires a database name (db_name) and uses the imported username,
    hostname, and password from an env file. 
    A url string is returned using the format required to connect to a SQL server.
    '''
    url = f'mysql+pymysql://{username}:{password}@{host}/{db_name}'
    return url

def get_telco_data(use_cache = True):
    '''
    This function is used to acquire the Telco dataset from the SQL server. It has no 
    required inputs, and checks the cache to see if the requested data already exists locally.
    Creates a dataframe from the SQL query, and then uses that dataframe to create a csv file.
    Returns the dataframe that is created.
    '''
    # Checking to see if data already exists in local csv file
    if os.path.exists('telco.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('telco.csv')
    # If data is not local we will acquire it from SQL server
    print('Acquiring data from SQL db')
    # Query to refine what data we want to grab (all of it mostly)
    query = '''
    SELECT * 
    FROM customers
    JOIN internet_service_types USING (internet_service_type_id)
    JOIN contract_types USING (contract_type_id)
    JOIN payment_types USING (payment_type_id)
    '''
    # Command line interaction with SQL server and assignment to dataframe (df)
    df = pd.read_sql(query, get_db_url('telco_churn'))
    # Creation of csv file
    df.to_csv('telco.csv', index=False)
    # Returns the dataframe
    return df

def prep_telco(df):
        '''
        Takes a dataframe as an argument and does the following:
        - Drops columns: 'customer_id', 'internet_service_type_id', 'contract_type_id', payment_type_id'
        - Modifies total_charges column to address empty value issue and change to proper type
            - Drops the NaN values (that exist in total_charges)
        - Encodes categorical columns and concats them to the df
        Splits the data into train, validate, test using telco_split()
        Returns train, validate, test (dataframes)
        '''
        # Drop any duplicates
        df.drop_duplicates(inplace=True)
        # Drop duplicated columns
        df = df.drop(columns=['internet_service_type_id', 'contract_type_id', 'payment_type_id'])
        # Replaces empty string total_charges as NaN values and converts the column to float
        df.total_charges = df.total_charges.replace(' ', np.nan).astype(float)
        # Drop the NaN values
        df.dropna(inplace=True)
        # Create a list of categorical columns
        cat_cols = [col for col in df.columns if df[col].dtype == 'O']
        # We want to retain customer_id without encoding it across data
        cat_cols.remove('customer_id')
        # Iterate through the categorical columns to encode them
        for col in cat_cols:
            dummy_df = pd.get_dummies(df[col],
                            prefix = df[col].name,
                            drop_first=True,
                            dummy_na=False)
            # Add the dummy vars to the df
            df = pd.concat([df, dummy_df], axis=1)
            # Delete the original (non-encoded) columns
            df = df.drop(columns=col)
        # Split our df in to train, validate, and test splits
        train, validate, test = telco_split(df)
        return train, validate, test
    
def telco_split(df):
    '''
    This function takes in a dataframe and returns train, validate, test splits. (dataframes)
    An initial 20% of data is split to place as 'test'
    A second split is performed (on the remaining 80%) between train and validate (70/30)
    '''
    # First split with 20% going to test
    train_validate, test = train_test_split(df, train_size = .8, 
                                                stratify= df.churn_Yes, random_state = 123)
    # Second split with 70% of remainder going to train, 30% to validate
    train, validate = train_test_split(train_validate, train_size = .7,
                                                stratify= train_validate.churn_Yes, random_state=123)
    # Return train, validate, test (56%, 24%, 20% splits of original df)
    return train, validate, test


def wrangle_telco():
    '''
    This function uses the acquire, prepare, and split functions to return train, validate, and test splits of Telco data
    '''
    df = get_telco_data()
    train, validate, test = prep_telco(df)
    return train, validate, test


def plot_variable_pairs(df):
	sns.pairplot(df, kind='reg', diag_kind='hist')
	return plt.show()

def months_to_years(df):
    df['tenure_years'] = round(df['tenure'] / 12).astype('int')
    return df

def plot_categorical_and_continuous_vars(df, conts, cats):
    for col in conts:
        sns.displot(x=col, data=df, kind='kde')
        plt.show()
        sns.boxplot(df[col])
        plt.show()
        sns.histplot(df[col])
        plt.show()
    for col in cats:
        sns.countplot(x=col, data=df)
        plt.show()
