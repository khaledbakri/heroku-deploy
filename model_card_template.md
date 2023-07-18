# Model Card

## Model Details

Khaled Bakri created the model. It is logistic regression using the default hyperparameters in scikit-learn 1.3.0. The model task is to classify whether income exceeds $50K/yr based on census data.

## Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). The original data set has 32536 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics
The model was evaluated using precision score, recall score, and fbeta score, which values are 0.74, 0.27, and 0.4, respectively
