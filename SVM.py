import numpy as np
import pandas as pd
import pylab as pl

from sklearn.model_selection import train_test_split, cross_val_score , GridSearchCV , StratifiedShuffleSplit
from sklearn.svm import SVC


if __name__ == "__main__":

    feature_vectors = pd.read_csv('galaxy_feature_vectors.csv', delimiter=',', header=None)

    X_galaxy = pd.read_csv('galaxy_feature_vectors.csv', delimiter=',', header=None).values[:, 0:-1]
    Y_galaxy = pd.read_csv('galaxy_feature_vectors.csv', delimiter=',', header=None).values[:, -1:].astype(int).flatten()

    X_train, X_test, Y_train, Y_test = train_test_split(X_galaxy, Y_galaxy, test_size=0.20, random_state=42,stratify=Y_galaxy)

    jobs = 6
    cache_size = 2048
    k = 5

    cv = StratifiedShuffleSplit(n_splits=k, test_size=0.2, random_state=42)
    svc = SVC(cache_size=cache_size)

    #param_grid_linear = {'kernel': ['linear'], 'C': [10 ** (-3), 10 ** (-1), 1, 10], 'class_weight': ['balanced'],
    #                     'gamma': ['scale']}

    param_grid_linear = {'kernel': ['linear'], 'C': [10 ** (-1), 1, 10], 'class_weight': ['balanced'],
                         'gamma': ['scale']}

    grid_linear = GridSearchCV(svc, param_grid=param_grid_linear, cv=cv, n_jobs=-1, scoring='accuracy', verbose=4)
    grid_linear.fit(X_train, Y_train)

    print("LINEAR : The best hyperparameters are %s with a score of %0.2f" % (
    grid_linear.best_params_, grid_linear.best_score_))

    #param_grid_rbf = {'kernel': ['rbf'], 'C': [10 ** (-3), 10 ** (-1), 1, 10], 'gamma': [10 ** (-3), 10 ** (-1), 1, 10]}
    #grid_rbf = GridSearchCV(svc, param_grid=param_grid_rbf, cv=cv, n_jobs=jobs, scoring='accuracy', verbose=4)

    #grid_rbf.fit(X_train, Y_train)

    print("LIN : The best hyperparameters are %s with a score of %0.2f" % (grid_linear.best_params_, grid_linear.best_score_))
    print('{0}'.format(grid_linear.cv_results_))
    # plot the scores of the grid
    # grid_scores_ contains parameter settings and scores
    C_range=param_grid_linear['C']
    gamma_range=param_grid_linear['gamma']

    grid=grid_linear
    score_dict = grid.cv_results_

    # We extract just the scores
    scores = [x[1] for x in score_dict]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))

    # Make a nice figure
    pl.figure(figsize=(8, 6))
    pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
    pl.xlabel('gamma')
    pl.ylabel('C')
    pl.colorbar()
    pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    pl.yticks(np.arange(len(C_range)), C_range)
    pl.show()