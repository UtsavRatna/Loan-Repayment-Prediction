import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train():
    df = pd.read_csv('Loan-data.csv')[:23000]
    df1 = df.drop(0,axis=0)
    df2 = df1.drop('Unnamed: 0',axis=1)
    train, test = train_test_split(df2, test_size=0.2)
    X_train = train.drop('Y',axis=1)
    y_train = train['Y']
    X_test = test.drop('Y',axis=1)
    y_test = test['Y']
    
    model = LogisticRegression()
    model.fit(X_train,y_train)
    

if __name__ = "__main__":
    train()