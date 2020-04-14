import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load csv files of messages and categories
    Merge the two datasets into one panda dataframe

    INPUT
        messages_filepath (str): file path to the messages file
        categories_filepath (str): file path to the categories file
    OUTPUT
        df (DataFrame) : dataset merged from messages and categories
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    '''
    Clean the merged dataframe from messages and categories

    INPUT
        df (DataFrame): dataset before cleaning
    OUTPUT
        df (DataFrame): dataset after cleaning
    '''

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = [x[0:-2] for x in row]
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str.get(-1).astype(int)

    # drop the original categories column from `df`
    df.drop(axis=1, columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    Save the cleaned dataset into a SQLite database

    INPUT
        df (DataFrame): cleaned dataset
    '''

    engine = create_engine("sqlite:///"+ database_filename)
    df.to_sql('messages', engine, if_exists='replace', index=False)



def main():
    """
    Main Data Processing function
    This function implement the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and pre-processing
        3) Data loading to SQLite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
