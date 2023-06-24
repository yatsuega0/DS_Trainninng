import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report


class Create_Dataset():
    def split_data(self, df):
        """ Split data 
            Training data and validation data　"""
        train = df[df['Survived'].notna()]
        test = df[df['Survived'].isna()].drop('Survived', axis=1)

        """ to numpy_array """
        X = train.values[:,1:]  
        y = train.values[:,0] 
        test_X = test.values

        return X, y, test_X
 

    def split_train_test(self, X, y):
        """ Hold-Out """
        test_size = 0.2
        random_state = 0
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return X_train, X_test, y_train, y_test
    
    
class Model_Trial():
    def model_svc(self, X_train, y_train):
        """ Support Vector Machine """
        svc = SVC(random_state=0)
        svc.fit(X_train, y_train)
        
        return svc
    
    def model_validate(self, model, X_test, y_test):
        """ validation """
        y_pred =  model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cl_report = classification_report(y_test, y_pred)
        
        return accuracy, cl_report

    
class Model_Output():
    def __init__(self):
        self.param_grid = None
        self.cv = None
        self.best_estimator_ = None
    
    def set_param_grid(self, param_grid):
        self.param_grid = param_grid
    
    def set_cv(self, n_splits, n_repeats, random_state):
        self.cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    def parameter_tuning_fit(self, X, y):
        """ Hyperparameter tuning GridSearch 
            train the model """
        model = SVC(random_state=0)
        grid_search = GridSearchCV(estimator=model, param_grid=self.param_grid, cv=self.cv)
        grid_search.fit(X, y)
        
        self.best_estimator_ = grid_search.best_estimator_
        
        
    def predict(self, test_X):
        """ inference on test data """
        if self.best_estimator_ is None:
            raise ValueError("Fit the model before making predictions.")
        
        return self.best_estimator_.predict(test_X)
    
    
    def result_to_csv(self, test_data, y_pred):
        """ output to csv """
        PassengerId = test_data['PassengerId']
        predictions = y_pred
        submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
        return submission.to_csv("result.csv", index=False)