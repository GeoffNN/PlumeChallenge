import numpy as np
import pandas as pd
from pandas import read_csv

def read_file_train():
    return read_csv("../data/X_train.csv")


def read_file_test():
    return np.genfromtxt("../data/X_test.csv", delimiter=',')

X_train = read_file_train()
