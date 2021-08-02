import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Text, Dict

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class UnsupportedClassifier(Exception):
    def __init__(self, estimator_name):
        self.msg= f"Unsupported estimator {estimator_name}"
        super().__init__(self.msg)


def get_supported_estimator() -> Dict:
    return {
        'logreg': LogisticRegression,
        'svm': SVC,
        'knn': KNeighborsClassifier
    }


def train_lr(df= pd.DataFrame, target_column= Text) -> LogisticRegression:
    y_train = df.loc[:, 'target'].values.astype('int32')
    X_train = df.drop('target', axis=1).values.astype('float32')

    logreg = LogisticRegression(C=0.001, solver='lbfgs', multi_class='multinomial', max_iter=100)
    logreg.fit(X_train, y_train)

    return logreg