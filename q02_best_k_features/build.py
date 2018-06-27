# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df, k=20):
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]
    
    selector=SelectPercentile(f_regression,k)
    X_new=selector.fit_transform(X,y)
    names=X.columns.values[selector.get_support()]    
    scores = selector.scores_[selector.get_support()]
    
    names_scores = list(zip(names,scores))
    
    sorted_list  = sorted(names_scores, key = lambda x : x[1], reverse = True)
    sorted_names =  [i[0] for i in sorted_list]
    
    return sorted_names
    
percentile_k_features(data)


