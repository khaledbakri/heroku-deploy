import pytest
import pandas as pd
from os.path import exists

def test_import_data():
    """ Test the import data."""
    try:
        pd.read_csv("./data/census.csv")
    except:
        print("The data is missing or the path is wrong.")

def test_data_shape():
    """ If your data is assumed to have no null values then this is a valid test. """
    df = pd.read_csv("./data/census.csv")
    assert df.shape == df.dropna().shape, "Dropping null changes shape."

def test_train_model():
    """ test if the model was trained and saved"""
    assert exists("./model/model.joblib")

