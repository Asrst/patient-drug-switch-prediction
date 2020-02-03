#!/usr/bin/env python
# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
from scipy import sparse 
from scipy.sparse import hstack
import random
import time
import os


### Utils:

def create_num_id(df):
    """
    create an unique numerical index for patients
    """
    df['id'] = df['patient_id'].apply(lambda x:int(x.split('_')[1]))
    return df

def sort_data(df, col_order=["id", 'event_name', 'specialty', 'plan_type']):
    """
    to sort the data in the predefined order
    """
    df.sort_values(col_order, inplace = True)
    df.reset_index(drop=1, inplace=True)
    return df

def get_sparse_matrix(data):
    return sparse.csr_matrix(data)

def sparse_vars(a, axis=None):
    """ 
    Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))

def sparse_stds(a, axis=None):
    """
    Standard deviation of sparse matrix a
    std = sqrt(var(a))
    """
    return np.sqrt(sparse_vars(a, axis))

# to get feature stats
def create_fitness_stats(df, cols, pos_idx, neg_idx, nans = True):
    
    stat_df = pd.DataFrame(data = cols, columns=['feature_name'])
    
    if nans:
        ###
        stat_df['avg_0'] = np.nanmean(df[neg_idx,:].astype(float), axis = 0)
        stat_df['avg_1'] = np.nanmean(df[pos_idx,:].astype(float), axis = 0)
        ###
        stat_df['sd_0'] = np.nanstd(df[neg_idx,:].astype(float), axis = 0)
        stat_df['sd_1'] = np.nanstd(df[pos_idx,:].astype(float), axis = 0)
        
    if not nans:
        ###
        stat_df['avg_0'] = np.ravel(df[neg_idx,:].mean(axis = 0))
        stat_df['avg_1'] = np.ravel(df[pos_idx,:].mean(axis = 0))
        ###
        stat_df['sd_0'] = np.ravel(sparse_stds(df[neg_idx,:], axis = 0))
        stat_df['sd_1'] = np.ravel(sparse_stds(df[pos_idx,:], axis = 0))
        
    return stat_df


# ### recency feature

# In[3]:


def create_recency_feature(df):
    start = time.time()
    cdfs = []
    for col in ['event_name', 'specialty', 'plan_type']:
        cat_df = df.groupby(["id", col]).agg({"event_time":np.min}).unstack(level=col)
        cat_df.columns = ['__'.join(['recency', col, name,]) for name in cat_df.columns.droplevel()]
        cdfs.append(cat_df)
    res_df = pd.concat(cdfs, axis = 1)
    # res_df = res_df.fillna('##')
    end = time.time()
    print('time taken (in secs) for recency features creation:', end-start)
    
    res_idx, res_col = np.array(res_df.index), np.array(res_df.columns)
    res_data = res_df.values

    del res_df
    return res_idx, res_col, res_data


# ### frequency feature

# In[4]:


def create_frequency_feature(temp_df):
    """
    function to create frequency feature 
    """
    start = time.time()
    cat_dfs = []
    for num in np.arange(1080,0,-30):
        temp_df.loc[temp_df['event_time'] > int(num), 'event_time'] = np.nan
        for col in ['event_name', 'specialty', 'plan_type']:
            cat_df = temp_df.groupby(["id", col],).agg({"event_time": 'count'}).unstack(level=col)
            cat_df.columns = ['__'.join(['frequency', col, name, str(int(num))]) for name in cat_df.columns.droplevel()]
            cat_dfs.append(cat_df)
    res_df = pd.concat(cat_dfs, axis = 1)
    res_df = res_df.fillna(0)
    end = time.time()
    print('time taken (in secs) for frequency feature creation:', end-start)
    
    res_idx, res_col = np.array(res_df.index), np.array(res_df.columns)
    res_data = get_sparse_matrix(res_df.values)
    
    del res_df
    # get data
    return res_idx, res_col, res_data


# ### NormChange

# In[5]:


def get_post_df(temp_post_df):
    """
    function to create feature matrix greather than time period for comparison

    """
    
    cat_dfs = []
    for num in np.arange(1080/2,0,-30):
        # making > null i.e keeping <=
        temp_post_df.loc[temp_post_df['event_time'] > int(num), 'event_time'] = np.nan
        for col in ['event_name', 'specialty', 'plan_type']:
            cat_df = temp_post_df.groupby(["id", col]).agg({"event_time": 'count'}).unstack(level=col)
            cat_df = cat_df/num
            cat_df.columns = ['__'.join(['normChange', col, name, str(int(num))]) for name in cat_df.columns.droplevel()]
            cat_dfs.append(cat_df)  
    post_df = pd.concat(cat_dfs, axis = 1)
    return post_df.fillna(0)


def get_pre_df(temp_pre_df):
    """
    function to create feature matrix less than time period for comparison
    """
    
    event_time_max = temp_pre_df['event_time'].max()
    cat_dfs = []
    for num in np.arange(0,(1080/2)+1,30)[1:]:
        # making <= null i.e keeping >
        temp_pre_df.loc[temp_pre_df['event_time'] <= int(num), 'event_time'] = np.nan
        for col in ['event_name', 'specialty', 'plan_type']:
            cat_df = temp_pre_df.groupby(["id", col]).agg({"event_time": 'count'}).unstack(level=col)
            cat_df = cat_df/(event_time_max-num)
            cat_df.columns = ['__'.join(['normChange', col, name, str(int(num))]) for name in cat_df.columns.droplevel()]
            cat_dfs.append(cat_df)
    pre_df = pd.concat(cat_dfs, axis = 1)        
    return pre_df.fillna(0)


def create_norm_feature(temp_df):
    """
    function to create norm change feature
    """
    
    start = time.time()
    
    post_df = get_post_df(temp_df)
    pre_df = get_pre_df(temp_df)
    
    res_col = np.array(pre_df.columns)
    post_df = post_df[res_col]
    r = np.where(post_df > pre_df, 1, 0)
    
    res_idx = np.array(post_df.index)
    res_data = get_sparse_matrix(r)
    
    end = time.time()
    print('time taken (in secs) for norm change feature creation:', end-start)
    
    # df1.where(df1.values==df2.values)
    # post_df.where(post_df > pre_df, 1, 0, inplace = True)
    del post_df, pre_df
    return res_idx, res_col, res_data


# ### transform & save train/test data

# In[6]:


def transform_data(data_df, target_df = None):
    """
    function to transform given matrix into feature matrix
    """
    rec_idx, rec_col, rec_data = create_recency_feature(data_df)
    freq_idx, freq_col, freq_data = create_frequency_feature(data_df)
    norm_idx, norm_col, norm_data = create_norm_feature(data_df)

    # with hstack function we are concatinating a sparse matrix and a dense matirx :)
    feat_df = hstack((rec_data, freq_data, norm_data))
    print('Final feature matrix shape:', feat_df.shape)
    
    # merge all the feature names
    feat_names = list(rec_col) + list(freq_col) + list(norm_col)
    
    if isinstance(target_df, pd.core.frame.DataFrame):
        # get +ve & -ve indices
        one_idx = target_df[target_df['outcome_flag'] == 1]['id'].index.tolist()
        zero_idx = target_df[target_df['outcome_flag'] == 0]['id'].index.tolist()
        
        # calculate fitness values of features
        rcdf = create_fitness_stats(rec_data, rec_col, one_idx, zero_idx, nans = True)
        fqdf = create_fitness_stats(freq_data, freq_col, one_idx, zero_idx, nans = False)
        nrdf = create_fitness_stats(norm_data, norm_col, one_idx, zero_idx, nans=False)
        fit_df = rcdf.append(fqdf).append(nrdf)
        fit_df.reset_index(drop=1)
        return feat_df, feat_names, fit_df
    
    return feat_df, feat_names

# ### fitness calculation

def fitness_calculation(data):
    if ((data['sd_0'] == 0 ) and (data['sd_1'] == 0)) and (((data['avg_0'] == 0) and (data['avg_1'] != 0)) or ((data['avg_0'] != 0) and (data['avg_1'] == 0))):
        return 9999999999
    elif (((data['sd_0'] == 0 ) and (data['sd_1'] != 0)) or ((data['sd_0'] != 0) and (data['sd_1'] == 0))) and (data['avg_0'] == data['avg_1']):
        return 1
    elif ((data['sd_0'] != 0 ) and (data['sd_1'] != 0)) and (data['avg_0'] != 0):
        return ((data['avg_1']/data['sd_1'])/(data['avg_0']/data['sd_0']))
    elif ((data['sd_0'] != 0 ) and (data['sd_1'] != 0)) and ((data['avg_0'] == 0) and (data['avg_1'] != 0)):
        return 9999999999
    else:
        return 1

# function to load & save sparse matrices
def save_data(data, path):
    return sparse.save_npz(path, data)

def load_data(path):
    return sparse.load_npz(path)


if __name__ == "__main__":
    
    data_paths = {}
    for dirname, _, filenames in os.walk('data/'):
        for filename in filenames:
            data_paths[filename] = os.path.join(dirname, filename)

    # preprocess target 
    target_df = pd.read_csv(data_paths['train_labels.csv'])
    print('train labels csv: ', target_df.shape, target_df.columns)
    target_df = create_num_id(target_df)
    target_df = sort_data(target_df, col_order=['id'])
    target_df.to_parquet('data/train_labels.parquet', index = False)

    # preprocess train data
    train_df = pd.read_csv(data_paths['train_data.csv'])
    print('train data csv: ', train_df.shape, train_df.columns)
    train_df = create_num_id(train_df)
    train_df = sort_data(train_df)


    # preprocess test data
    test_df = pd.read_csv(data_paths['test_data.csv'])
    print('test data csv: ', test_df.shape, test_df.columns)
    test_df = create_num_id(test_df)


    # In[9]:
    cat_columns = ['event_name', 'specialty', 'plan_type']
    train_unique_col_values = {col:train_df[col].unique() for col in cat_columns}
    test_unique_col_values = {col:test_df[col].unique() for col in cat_columns}

    # train missed
    train_missed = {k:[] for k in cat_columns}
    for col in cat_columns:
        for col_type in test_unique_col_values[col]:
            if col_type not in train_unique_col_values[col]:
                train_missed[col].append(col_type)
        print('missing types in train', col, len(train_missed[col]))

    # missed values
    test_missed = {k:[] for k in cat_columns}
    for col in cat_columns:
        for col_type in train_unique_col_values[col]:
            if col_type not in test_unique_col_values[col]:
                test_missed[col].append(col_type)
        print('missing types in test', col, len(test_missed[col]))


    # get all patient test ids
    test_patient_ids = test_df['patient_id'].values
    # create duplicate values for missing data
    dup_values = []
    for col in cat_columns:
        iter_items = test_missed[col]
        if len(iter_items) > 0:
            for item in iter_items:
                rc = random.choice(test_patient_ids)
                et = np.nan
                pa = np.nan
                if col == 'event_name':
                    en = item
                    sn = random.choice(train_unique_col_values['specialty'])
                    pt = random.choice(train_unique_col_values['plan_type'])
                if col == 'specialty':
                    en = random.choice(train_unique_col_values['event_name'])
                    sn = item
                    pt = random.choice(train_unique_col_values['plan_type'])     
                if col == 'plan_type':
                    en = random.choice(train_unique_col_values['event_name'])
                    sn = random.choice(train_unique_col_values['specialty'])
                    pt = item
                dup_values.append([rc, en, et, sn, pt, pa, int(rc.split('_')[1])])


    # In[11]:
    test_df = test_df[~(test_df['specialty'].isin(train_missed['specialty']))].reset_index(drop = 1)


    # In[12]:
    dup_df = pd.DataFrame(data = dup_values, columns = test_df.columns)
    dup_df.shape, dup_df.columns


    # In[13]:
    test_df = test_df.append(dup_df)
    test_df = sort_data(test_df)
    test_df.shape, test_df.columns


    # In[14]:
    train_df.to_parquet('data/train.parquet', index = False)
    test_df.to_parquet('data/test.parquet', index = False)


    ## update files in the data path
    for dirname, _, filenames in os.walk('data/'):
        for filename in filenames:
            data_paths[filename] = os.path.join(dirname, filename)
    
    # read train & target parquet data files
    train_df = pd.read_parquet(data_paths['train.parquet'])
    print('train data:', train_df.shape, train_df.columns)

    target_df = pd.read_parquet(data_paths['train_labels.parquet'])
    print('train labels', target_df.shape, target_df.columns)
    
    # create train features:
    train_feat, train_feat_names, train_fit_df = transform_data(train_df, target_df)
    train_feat.shape, len(train_feat_names)


    # save & delete train data:
    save_data(train_feat, 'data/train_features.npz')
    del train_df, target_df
    
    # train_fitness_val 
    train_fit_df['fitness_value'] = train_fit_df.apply(fitness_calculation, axis = 1)
    train_fit_df.to_csv('data/train_fitness_values.csv', index = None)


    # read test data
    test_df = pd.read_parquet(data_paths['test.parquet'])
    print(test_df.shape, test_df.columns)


    # create test features & save
    test_feat, test_feat_names = transform_data(test_df)
    test_feat.shape, len(test_feat_names)
    save_data(test_feat, 'data/test_features.npz')
    del test_df


    # save train & test features
    # assert(train_feat_names == test_feat_names)
    pd.DataFrame(train_feat_names, columns = ['feature']).to_csv('data/train_feature_names.csv', index = False)
    pd.DataFrame(test_feat_names, columns = ['feature']).to_csv('data/test_feature_names.csv', index = False)