import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression

def read_file_train():
    return read_csv("../data/X_train.csv", index_col=0)

def read_file_train_output():
    return read_csv("../data/challenge_output_data_training_file_predict_air_quality_at_the_street_level.csv", index_col=0)


def read_file_test():
    return read_csv("../data/X_test.csv", index_col=0)

def set_buffer_nans_to_zero(data):
    data.fillna(0, inplace=True)



def triang_correlation_matrix(d, title):
    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    h = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.xticks(rotation=35)
    plt.yticks(rotation=35)
    plt.title(title)


pd.set_option('display.max_columns', 33)
X_train = read_file_train()
X_test = read_file_test()
Y_train = read_file_train_output()
# X_train = pd.concat([X_train, Y_train], axis=1)
set_buffer_nans_to_zero(X_train)
set_buffer_nans_to_zero(X_test)

# Change categorical variables to numerical
cat_columns = ['zone_id']
cats_to_drop = ['station_id']
X_train = pd.get_dummies(X_train, columns=cat_columns)
X_test = pd.get_dummies(X_test, columns=cat_columns)
group_train = X_train.groupby(['zone_id'])
group_test = X_test.groupby(['zone_id'])

lin_regressor = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

lin_models = {}
lin_models_score = {}
y_pred = {}
scores = {}

for name, data in group_train:
    idx = data.index
    lin_regressor = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=4)
    lin_models[name] = lin_regressor.fit(X_train.loc[idx].drop('pollutant', axis=1), Y_train.loc[idx])
    lin_models_score[name] = lin_regressor.score(X_train.loc[idx].drop('pollutant', axis=1), Y_train.loc[idx])
    scores[name] = cross_val_score(lin_models[name], X_train.loc[idx].drop('pollutant', axis=1), Y_train.loc[idx], cv=5)

for name, data in group_test:
    idx = data.index
    y_pred[name] = lin_models[name].predict(X_test.loc[idx].drop('pollutant', axis=1))




