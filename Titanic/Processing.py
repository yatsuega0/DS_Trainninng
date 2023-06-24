import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class Get_Data():
    def read_data(self):
        train_data = pd.read_csv('train.csv')
        
        test_data = pd.read_csv('test.csv')
        
        test_data['Survived'] = np.nan
        df = pd.concat([train_data.copy(), test_data.copy()], sort=False, ignore_index=True, axis=0)
        
        return test_data, df
    
    
    def missing_value(self, df):
        
        """ Age """
        age_ = df[['Age','Pclass','Sex','Parch','SibSp']]
        age_ = pd.get_dummies(age_)

        known_age = age_[age_['Age'].notna()].values  
        unknown_age = age_[age_['Age'].isna()].values

        X = known_age[:, 1:]  
        y = known_age[:, 0]

        rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
        rfr.fit(X, y)

        pred_Age = rfr.predict(unknown_age[:, 1::])
        df.loc[df['Age'].isna(), 'Age'] = pred_Age
        
        
        """ Embarked """
        df['Embarked'] = df['Embarked'].fillna('S') 
        
        """ Fare """
        fare = df.loc[(df['Embarked'] == 'S') & (df['Pclass'] == 3), 'Fare'].median()
        df['Fare'] = df['Fare'].fillna(fare)
        
        """ Cabin """
        df['Cabin'] = df['Cabin'].fillna('Unknown')
        df['Cabin_label'] = df['Cabin'].str.get(0)
        
        return df 
    
    
    def feature_and_processing(self, df):
        """ Sex """
        sex_mapping = {'male': 0, 'female': 1}
        df['Sex'] = df['Sex'].map(sex_mapping).astype(int)
        
        """ extract title from name """
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 
                                         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df['Title'] = df['Title'].map(title_mapping)
        df['Title'] = df['Title'].fillna(0)
        
        """ unnecessary columns drop """
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        
        """ One-Hot-encoding """
        df = pd.get_dummies(df)
        
        """ StandardScaler """
        scaler = StandardScaler()
        select = ['Age', 'Fare']
        df[select] = scaler.fit_transform(df[select])
    
        return df
