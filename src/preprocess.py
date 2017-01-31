import numpy as np
import pandas as pd
from pandas import read_csv

def read_file_train():
    return read_csv("../data/X_train.csv", index_col=0)


def read_file_test():
    return read_csv("../data/X_test.csv", index_col=0)

def set_buffer_nans_to_zero(data):
    data.fillna(0, inplace=True)

def set_buffer_nans_to_mean(data):
    pass


def preprocess_pipeline(data):
    set_buffer_nans_to_zero(data)
    cat_columns =  ['zone_id', 'station_id', 'pollutant']

pd.set_option('display.max_columns',33)
X_train = read_file_train()
X_test = read_file_test()
set_buffer_nans_to_zero(X_train)
set_buffer_nans_to_zero(X_test)

cat_columns = ['zone_id', 'station_id', 'pollutant']
d_train = pd.get_dummies(X_train, columns=cat_columns)
d_test = pd.get_dummies(X_test, columns=cat_columns)

X_train = pd.concat([X_train.drop(cat_columns, axis=1), d_train], axis=1)
X_test = pd.concat([X_test.drop(cat_columns, axis=1), d_test], axis=1)

