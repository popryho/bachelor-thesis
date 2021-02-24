from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from main import plot_decision_boundary


if __name__ == '__main__':

    X, y = make_moons(n_samples=100,
                      shuffle=True,
                      noise=0.05,
                      random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=2, random_state=42, stratify=y
    )

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, -1 * np.ones_like(y_test)], axis=0)

    lblprop = LabelPropagation()
    lblprop.fit(X, y)
    y_pred_prop = lblprop.predict(X[y == -1])

    sns.heatmap(confusion_matrix(y_test, y_pred_prop), annot=True, fmt="d", cmap='Blues')
    print("Accuracy_score: ", accuracy_score(y_test, y_pred_prop))
    plt.show()

    plot_decision_boundary(X, y, y_test, y_pred_prop, lblprop)
    plt.show()

    lblsprd = LabelSpreading()
    lblsprd.fit(X, y)
    y_pred_sprd = lblsprd.predict(X[y == -1])

    sns.heatmap(confusion_matrix(y_test, y_pred_sprd), annot=True, fmt="d", cmap='Greens')
    print("Accuracy_score: ", accuracy_score(y_test, y_pred_sprd))
    plt.show()

    plot_decision_boundary(X, y, y_test, y_pred_sprd, lblsprd)
    plt.show()
