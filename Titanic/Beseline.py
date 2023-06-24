import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def create_data(train, test):
    y_col = 'Survived'
    X = train[['Pclass', 'Parch']].values
    y = train[y_col]
    test_X = test[['Pclass', 'Parch']].values
    
    return X, y, test_X


def split_train_test(X, y, test_size=0.2, random_state=0):
    """ Hold-Out """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
    
def model_svc(X, y, test_X, random_state=0):
    svc = SVC(random_state=random_state)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(test_X)

    return y_pred


def result_to_csv(test_data, y_pred):
    PassengerId = test_data['PassengerId']
    predictions = y_pred
    submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
    return submission.to_csv("result.csv", index=False)


if __name__ == '__main__' :
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    display(train_data.head())
    display(test_data.head())
    X, y, test_X = create_data(train_data, test_data)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    y_pred =  model_svc(X_train, y_train, X_test)
    print(accuracy_score(y_test, y_pred))
    y_pred =  model_svc(X, y, test_X)
    result_to_csv(test_data, y_pred)
    
    
    
    
    