import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Text, Dict

from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
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


def train(df= pd.DataFrame, target_column= Text,
          estimator_name= Text, param_grid= Dict, cv= int):
    estimators= get_supported_estimator()
    if estimator_name not in estimators.keys():
        raise UnsupportedClassifier(estimator_name)

    estimator= estimators[estimator_name]()
    fi_scorer= make_scorer(f1_score, average= 'weighted')
    clf= GridSearchCV(estimator= estimator,
                      param_grid= param_grid,
                      cv= cv,
                      verbose= 1,
                      scoring= fi_scorer
                      )

    y_train= df.loc[:, target_column].values.astype('int32')
    x_train= df.drop(target_column, axis= 1).values.astype('float32')
    clf.fit(x_train, y_train)

    return clf


def train_lr(df= pd.DataFrame, target_column= Text) -> LogisticRegression:
    y_train = df.loc[:, 'target'].values.astype('int32')
    X_train = df.drop('target', axis=1).values.astype('float32')

    logreg = LogisticRegression(C=0.001, solver='lbfgs', multi_class='multinomial', max_iter=100)
    logreg.fit(X_train, y_train)

    return logreg