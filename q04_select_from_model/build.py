# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    model = RandomForestClassifier(random_state = 9)
    X,y = df.iloc[:,:-1], df.iloc[:,-1]
    sfm = SelectFromModel(model)
    X_new = sfm.fit(X,y)
    feature_name = list(X.columns.values[sfm.get_support()])
    
    return feature_name

select_from_model(data)
    

