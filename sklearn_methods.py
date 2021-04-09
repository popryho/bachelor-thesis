import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading,
    SelfTrainingClassifier
)
from sklearn.svm import SVC

from main import plot_decision_boundary, get_data


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, -1 * np.ones_like(y_test)], axis=0)

    models = (
        LabelPropagation(max_iter=10000),
        LabelSpreading(),
        SelfTrainingClassifier(base_estimator=SVC(probability=True, gamma="auto"))
    )
    color_maps = ('Blues', 'Greens', 'Reds')

    for model, cmap in zip(models, color_maps):
        model.fit(X, y)
        y_pred = model.predict(X[y == -1])

        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap=cmap)
        print('-'*50, f'\nModel name: {model.__str__()}\n'
              f'Accuracy_score: {accuracy_score(y_test, y_pred)}')
        plt.show()

        plot_decision_boundary(X, y, y_test, y_pred, model)
