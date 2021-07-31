import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.base import BaseEstimator
from typing import Text, Tuple


def evaluate(df= pd.DataFrame, target_column= Text, clf= BaseEstimator) -> Tuple[float, np.ndarray]:
    y_test = df.loc[:, 'target'].values.astype('int32')
    X_test = df.drop('target', axis=1).values.astype('float32')
    prediction = clf.predict(X_test)
    cm = confusion_matrix(prediction, y_test)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')

    return f1, cm