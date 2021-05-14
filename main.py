from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from Estimator import Estimator

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


def get_data(n_samples=100, noise=0.05):
    X_, y_ = make_moons(n_samples=n_samples,
                        noise=noise,
                        random_state=0)

    return train_test_split(
        X_, y_, train_size=2, random_state=42, stratify=y_
    )


def plot_dataset(X_, y_, c):
    plt.scatter(X_[y_ == -1][:, 0], X_[y_ == -1][:, 1], c=c, cmap='bwr', edgecolors='k')
    plt.scatter(X_[y_ != -1][:, 0], X_[y_ != -1][:, 1], c=y_[y_ != -1], cmap='bwr', edgecolors='k', marker='X', s=200)
    plt.show()


def plot_decision_boundary(X_, y_, y_test_, y_pred_, estimator):
    x1_min, x1_max = X_[y_ == -1][:, 0].min() - 0.5, X_[y_ == -1][:, 0].max() + 0.5
    x2_min, x2_max = X_[y_ == -1][:, 1].min() - 0.5, X_[y_ == -1][:, 1].max() + 0.5

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2),
                           np.arange(x2_min, x2_max, 0.2))

    Z = estimator.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.7, cmap='bwr')
    plot_dataset(X_, y_, y_pred_)
    plt.text(xx1.max() - .2, xx2.min() + .2, ('%.2f' % accuracy_score(y_test_, y_pred_)).lstrip('0'),
             size=15, horizontalalignment='right')


def download_dataset(X_, y_):
    dataset = pd.concat([pd.DataFrame(X_),
                         pd.DataFrame(y_.reshape(-1, 1))],
                        axis=1)
    dataset.to_csv('n=100, m=2.csv')


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = get_data()

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, -1 * np.ones_like(y_test)], axis=0)

    start_time = time()
    est = Estimator()
    est.fit(X, y)

    print(f'--- {time() - start_time} seconds ---')
    print(f'weights: {est.coeff_}, tolerance: {est.tol_}')

    start_time = time()
    y_pred = est.predict(X[y == -1])
    print(f'--- {time() - start_time} seconds ---')

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    print(f'Accuracy_score: {accuracy_score(y_test, y_pred)}')
    plt.show()

    plot_dataset(X, y, y_test)
    # plot_decision_boundary(X, y, y_test, y_pred, estimator=est)
