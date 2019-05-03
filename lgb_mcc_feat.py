# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:22:55 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:47:44 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:45:28 2019

@author: gungor2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
from lightgbm import LGBMModel,LGBMClassifier
from scipy import signal

from python_speech_features import mfcc
from python_speech_features import logfbank


np.random.seed(444)

train = pd.read_csv('./LANL-Earthquake-Prediction/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

rows = 150000
segments = int(np.floor(train.shape[0] / rows))

X_tr = pd.DataFrame(index=range(segments), dtype=np.float64)


y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

fs = 4*10**6
for segment in range(segments):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = pd.Series(seg['acoustic_data'].values)
    mfcc_feat = mfcc(x,fs)
    
    mfcc_feat = np.delete(mfcc_feat, 0, 0)
    print(mfcc_feat)
    mfcc_feat = mfcc_feat.reshape(-1)

#    fbank_feat = logfbank(x,fs)
#    
#    temp_vec = mfcc_feat.reshape(-1)
#    temp_vec.extend(fbank_feat.reshape(-1))
#    
    for j in range(len(mfcc_feat)):
        X_tr.loc[segment, 'm' + str(j)] = mfcc_feat[j]
        
        
    y = np.mean(seg['time_to_failure'].values)
    
    y_tr.loc[segment, 'time_to_failure'] = y
    
    
    
    
scaler = StandardScaler()
scaler.fit(X_tr)
X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)


submission = pd.read_csv('./LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)

for i, seg_id in enumerate((X_test.index)):
    seg = pd.read_csv('./LANL-Earthquake-Prediction/test/' + seg_id + '.csv')
    
    x = pd.Series(seg['acoustic_data'].values)
    
    mfcc_feat = mfcc(x,fs)
    
    mfcc_feat = np.delete(mfcc_feat, 0, 0)
    print(mfcc_feat)
    mfcc_feat = mfcc_feat.reshape(-1)

#    fbank_feat = logfbank(x,fs)
#    
#    temp_vec = mfcc_feat.reshape(-1)
#    temp_vec.extend(fbank_feat.reshape(-1))
#    
    for j in range(len(mfcc_feat)):
        X_test.loc[segment, 'm' + str(j)] = mfcc_feat[j]
        

    




        
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

oof = np.zeros(len(X_train_scaled))

prediction = np.zeros(len(X_test))

scores = []

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

def mean_absolute_error(x,y):
    diff = np.abs(x-y)
    return(diff)

params = {'num_leaves': 128,
          'min_data_in_leaf': 79,
          'objective': 'huber',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 1,
          "bagging_fraction": 0.75,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501
         }

for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train_scaled)):

    print(fold_n)
    X_train, X_valid = X_train_scaled.iloc[train_index], X_train_scaled.iloc[valid_index]
    y_train, y_valid = y_tr.iloc[train_index], y_tr.iloc[valid_index]
    model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1)
    model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=10000, early_stopping_rounds=200)
    y_pred_valid = model.predict(X_valid)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    
    oof[valid_index] = y_pred_valid
    scores.extend(mean_absolute_error(y_valid.values.reshape(-1), y_pred_valid))
    prediction += y_pred 
    
prediction /= n_fold
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

submission['time_to_failure'] = (prediction)
# submission['time_to_failure'] = prediction_lgb_stack
print(submission.head())
submission.to_csv('submission.csv')