# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    model = RandomForestClassifier()
    X,y = df.iloc[:,:-1], df.iloc[:,-1]
    rfe = RFE(model, X.shape[1]/2)
    rfe = rfe.fit(X,y)
    top_features = list(X.columns.values[rfe.support_])
    
    return top_features

rf_rfe(data)


