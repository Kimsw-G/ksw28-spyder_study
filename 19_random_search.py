from scipy.stats import uniform, randint
import numpy as np
rgen = randint(0,10)
print(rgen.rvs(10))

np.unique(rgen.rvs(1000),return_counts=True)
print(np.unique(rgen.rvs(1000),return_counts=True))

ugen = uniform(0,1)
print(ugen.rvs(10))

import numpy as np
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = \
    train_test_split(data,target,test_size=0.2,random_state=42)
    
sub_input, val_input, sub_target, val_target = \
    train_test_split(train_input,train_target,test_size=0.2,random_state=42)
    
def print_score(model,TEXT):
    print(f"###### {TEXT} #####")
    print(model.score(sub_input, sub_target))
    print(model.score(val_input, val_target))

params = {'min_impurity_decrease':uniform(0.0001,0.001),
          'max_depth': randint(20,50),
          'min_samples_split': randint(2,25),
          'min_samples_leaf': randint(1,25),}

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params,
                        n_iter=100,n_jobs=-1,random_state=42)
gs.fit(train_input, train_target)
print(gs.best_params_)

print(np.max(gs.cv_results_['mean_test_score']))

dt = gs.best_estimator_
print(dt.score(test_input, test_target))