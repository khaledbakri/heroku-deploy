# Script to train machine learning model.

import pandas as pd
import joblib
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def preprocess_data(X, encoder):
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    
    return X

def compute_metrics_on_slice_data(model, test):
    f = open("slice_output.txt", "w")
    education_categories = test["education"].unique()
    for education_category in education_categories:
        select_category = test["education"] == education_category
        X_filtered = test[select_category]
        X, y, _, _ = process_data(
            X_filtered, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        preds = inference(model, X)

        precision, recall, fbeta = compute_model_metrics(y, preds)
        f.write("Education category: %s - " % education_category)
        f.write("Precision: %.2f - " %  precision)
        f.write("Recall: %.2f - " %  recall)
        f.write("Fbeta: %.2f \n" %  fbeta)
    f.close()

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('../data/census.csv')
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=0)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Proces the train data with the process_data function
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Train and save a model.
    model = train_model(X_train, y_train)
    compute_metrics_on_slice_data(model, test)
    joblib.dump(model, '../model/model.joblib')
    joblib.dump(encoder, '../model/encoder.joblib')

