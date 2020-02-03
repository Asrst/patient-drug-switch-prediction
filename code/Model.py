#!/usr/bin/env python
# coding: utf-8

# In[1]:
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import sparse
import random
import time
import os

def save_data(data, path):
    return sparse.save_npz(path, data)

def load_data(path):
    return sparse.load_npz(path).tocsr()


if __name__ == "__main__":

    data_paths = {}
    for dirname, _, filenames in os.walk('/data'):
        for filename in filenames:
            data_paths[filename] = os.path.join(dirname, filename)
            print(os.path.join(dirname, filename))


    # read train df
    # train_df = pd.read_parquet(data_paths['train.parquet'])
    # print(train_df.shape, train_df.columns)


    # read target df
    target_df = pd.read_parquet(data_paths['train_labels.parquet'])
    print(target_df.shape, target_df.columns)


    # In[5]:
    train_features = load_data(data_paths['train_features.npz'])
    train_features.data = np.nan_to_num(train_features.data, copy=False)
    print('train features', train_features.shape)


    # In[6]:
    train_labels= target_df['outcome_flag'].values
    print('train labels', train_labels.shape)


    # In[7]:
    test_df = pd.read_parquet(data_paths['test.parquet'])
    # print(test_df.shape, test_df.columns)

    # In[8]:
    test_features = load_data(data_paths['test_features.npz'])
    test_features.data = np.nan_to_num(test_features.data, copy=False)
    print('test features', test_features.shape)


    ### Modelling - LGBM
    # In[9]:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    param = {
        'bagging_freq': 5,
        'bagging_fraction': 0.335,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.1,
        'learning_rate': 0.01,
        'max_depth': 97,
        'metric':'auc',
        'min_data_in_leaf': 61,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 19,
        'num_threads': 8,
        # 'is_unbalance': True,
        'scale_pos_weight': 11,
        'tree_learner': 'serial',
        'objective': 'binary',
        'verbosity': -1,
        'reg_alpha': 50,  # L1 weight reg
        'reg_lambda': 0,  # L2 weight reg
    }


    # In[10]:
    # imp_df = pd.read_csv(data_paths['kfold_feature_importance.csv'])
    # imp_df.sort_values(["importance"], inplace = True)
    # imp_df.reset_index(drop = 1)
    # imp_df['importance'].hist(bins = 50)
    # imp_feat_idx = sorted(imp_df[imp_df['importance'] > 0]['0'].tolist())
    # len(imp_feat_idx)


    # In[11]:
    # train_features = train_features[:,imp_feat_idx]
    # test_features = test_features[:,imp_feat_idx]
    # train_features.shape, test_features.shape


    # define kFold
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)

    # to store
    y_test_pred = np.zeros(test_features.shape[0])
    feature_importance_df = pd.DataFrame(np.arange(train_features.shape[1]))
    feature_importance_df["importance"] = 0

    # train, validate & predict on test
    start = time.time()
    for fold_, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_labels)):
        print("\nStarting Fold {}/{}".format(fold_+1, n_splits))

        trn_data = lgb.Dataset(train_features[trn_idx,:], label=train_labels[trn_idx])
        val_data = lgb.Dataset(train_features[val_idx,:], label=train_labels[val_idx])

        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, 
                        early_stopping_rounds = 250)

        feature_importance_df['importance'] += clf.feature_importance()/n_splits    
        y_test_pred += clf.predict(test_features, num_iteration=clf.best_iteration)/n_splits


    # create feature importance df
    feature_importance_df.sort_values(["importance"], inplace = True)
    feature_importance_df.reset_index(drop = 1)
    feature_importance_df.to_csv('data/kfold_feature_importance.csv', index = False)


    # create prediction df
    pred_df = pd.DataFrame(data = sorted(test_df['id'].unique()), columns = ['id'])
    pred_df['label'] = y_test_pred


    # create submission df
    sub_df = pd.read_csv(data_paths['Sample Submission.csv'])
    sub_df['id'] = sub_df['patient_id'].apply(lambda x:int(x.split('_')[1]))
    sub_df = sub_df.merge(pred_df, on='id', how='left')
    sub_df = sub_df.drop(['outcome_flag', 'id'], axis = 1)
    sub_df.columns = ['patient_id', 'outcome_flag']

    # save to fil
    filename = "BestScore.xlsx"
    sub_df.to_excel(filename, index = False)