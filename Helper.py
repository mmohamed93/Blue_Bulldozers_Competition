#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.lib.deepreload import reload as dreload
import PIL, os, numpy as np, math, collections, threading, json, bcolz, random, scipy, cv2
import pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time, shutil, copy
import seaborn as sns, matplotlib
import IPython, graphviz, sklearn_pandas, sklearn, warnings, pdb
import contextlib
from abc import abstractmethod
from glob import glob, iglob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import chain
from functools import partial
from collections import Iterable, Counter, OrderedDict
from isoweek import Week
from pandas_summary import DataFrameSummary
from IPython.lib.display import FileLink
from PIL import Image, ImageEnhance, ImageOps
from sklearn import metrics, ensemble, preprocessing
from operator import itemgetter, attrgetter
from pathlib import Path
from distutils.version import LooseVersion
from pdpbox import pdp
from plotnine import *

from matplotlib import pyplot as plt, rcParams, animation
from ipywidgets import interact, interactive, fixed, widgets
matplotlib.rc('animation', html='html5')
np.set_printoptions(precision=5, linewidth=110, suppress=True)

from ipykernel.kernelapp import IPKernelApp
def in_notebook(): return IPKernelApp.initialized()

def in_ipynb():
    try:
        cls = get_ipython().__class__.__name__
        return cls == 'ZMQInteractiveShell'
    except NameError:
        return False

import tqdm as tq
from tqdm import tqdm_notebook, tnrange

def clear_tqdm():
    inst = getattr(tq.tqdm, '_instances', None)
    if not inst: return
    try:
        for i in range(len(inst)): inst.pop().close()
    except Exception:
        pass

if in_notebook():
    def tqdm(*args, **kwargs):
        clear_tqdm()
        return tq.tqdm(*args, file=sys.stdout, **kwargs)
    def trange(*args, **kwargs):
        clear_tqdm()
        return tq.trange(*args, file=sys.stdout, **kwargs)
else:
    from tqdm import tqdm, trange
    tnrange=trange
    tqdm_notebook=tqdm

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz
from scipy.cluster import hierarchy as hc


# In[ ]:


def get_sample(df,n):
    """ Gets a random sample of n rows from df, without replacement.
    Parameters:
    -----------
    df: A pandas data frame, that you wish to sample from.
    n: The number of rows you wish to sample.
    Returns:
    --------
    return value: A random sample of n rows of df.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    >>> get_sample(df, 2)
       col1 col2
    1     2    b
    2     3    a
    """
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()


# In[3]:


def is_numeric(df,column=False):
    '''If a column is passed the function will return True if the column is numeric and False if not numeric.
    If no column is added the function will return a list with all numerical columns names in the dataframe.
    
    df: A pandas data frame, that you wish to sample from.
    column (default = False) : a column name in the dataframe'''
    numericals = []
    if column == False:
        for col in df.columns:
            if df[col].dtypes == 'int64' or df[col].dtypes == 'int32' or df[col].dtypes == 'float64':
                numericals.append(col)
        return numericals
    elif df[column].dtypes == 'int64' or df[column].dtypes == 'int32' or df[column].dtypes == 'float64':
        return True
    else:
        return False


# In[2]:


def is_categorical(df,column = False):
    '''If a column is passed the function will return True if the column is categorical and False if not.
    If no column is added the function will return a list with all categorical columns names in the dataframe.
    
    df: A pandas data frame, that you wish to sample from.
    column (default = False) : a column name in the dataframe'''
    categoricals = []
    if column == False:
        for col in df.columns:
            if df[col].dtypes == 'object' or df[col].dtype.name == 'category':
                categoricals.append(col)
        return categoricals
    elif df[column].dtypes == 'object' or df[column].dtype.name == 'category':
        return True
    else:
        return False


# In[5]:


def is_date(df,column = False):
    '''If a column is passed the function will return True if the column is a date and False if not.
    If no column is added the function will return a list with all date columns names in the dataframe.
    
    df: A pandas data frame, that you wish to sample from.
    column (default = False) : a column name in the dataframe'''
    dates = []
    if column == False:
        for col in df.columns:
            if not is_numeric(df,col) and not is_categorical(df,col):
                dates.append(col)
        return dates
    elif not is_numeric(df,column) and not is_categorical(df,column):
        return True
    else:
        return False


# In[6]:


def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):
    '''from fastai (old) the function will transfom data columns into many useful attributes
    df: A pandas data frame, that you wish to sample from.
    fldnames: date columns'''

    if isinstance(fldnames,str): 
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop: df.drop(fldname, axis=1, inplace=True)


# In[7]:


def dummy_df(df,to_dummy):
    '''The function will one hot encode all columns in a list in the dataframe
    
    df: A pandas data frame, that you wish to sample from.
    to_dummy: a list with column names that will be encoded'''
    for cat in to_dummy:
        dummy = pd.get_dummies(df[cat] , prefix = cat)
        df = df.drop(cat,1)
        df = pd.concat([df,dummy],axis = 1)
    return df


# In[8]:


def describe_categorical(df,sort_by = 0, report = True):
    '''The function will return a description of all categorical columns in a list (number of unique values
    and percentage of missing values) sorted by either missing values or unique values descendingly.
    
    df: A pandas data frame, that you wish to sample from.
    sort_by: use 0 for unique and 1 for missing (default = 0)
    report: will print a description (default = True)'''
    categoricals=is_categorical(df)
    missing =[]
    unique_list=[]
    for column in categoricals:
        unique_list.append(df[column].nunique())
    for column in categoricals:
        column_missing = (df[column].isnull().sum()/df.shape[0])*100
        missing.append(column_missing)
    if sort_by == 0:
        indices = np.argsort(unique_list)[::-1]
    else:
        indices = np.argsort(missing)[::-1]
    
    cat_missing_unique = []
    
    for i in range(len(categoricals)):
        nunique = df[categoricals[indices[i]]].nunique()
        column_missing = (df[categoricals[indices[i]]].isnull().sum()/df.shape[0])
        cat_missing_unique.append([categoricals[indices[i]],nunique,column_missing])
        if report == True:
            print('{} has {} values and {:.2f}% missing vlues'.format(categoricals[indices[i]],nunique,column_missing*100))
    return cat_missing_unique


# In[9]:


def LabelEncoder(df):
    cats = is_categorical(df)
    '''A robust label encoder, very good for tree algorithms and high cardinality features
    
    df: A pandas data frame, that you wish to sample from.
    cats: list of categorical values to be encoded '''
    for cat in cats:
        df[cat] = df[cat].astype('category')
        df[cat] = df[cat].cat.codes+1
    return df


# In[10]:


def drop_missing(df,threshold):
    '''the function will drop the columns which have a fraction of missing values above a certian threshold
    
    df: A pandas data frame, that you wish to sample from.
    threshold: a fraction of missing values to the total values'''
    for column in is_categorical(df):
        if df[column].isnull().sum()/df.shape[0] > threshold:
            df.drop(column,1,inplace = True)
    return df


# In[11]:


def display_all(df):
    '''fully display the output in a window '''
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[12]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m,X_train,X_test,y_train,y_test):
    '''enhanced score function, insert the fitted model in the function'''
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_test), y_test),
                m.score(X_train, y_train), m.score(X_test, y_test)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print('RMSE for training Data is {}'.format(res[0]))
    print('RMSE for test Data is {}'.format(res[1]))
    print('R2 for training Data is {}'.format(res[2]))
    print('R2 for test Data is {}'.format(res[3]))


# In[13]:


def fill_missing(df,strategy = 'median'):
    '''fill missing values in a given data set
    
    df: A pandas data frame, that you wish to sample from.
    strategy: median or mean or mode (default = median)'''
    numericals = is_numeric(df)
    if strategy == 'median':
        for i in numericals:
            if df[i].isnull().sum() != 0:
                df[i].fillna(df[i].median(),inplace = True)
    elif strategy == 'mean':
        for i in numericals:
            if df[i].isnull().sum() != 0:
                df[i].fillna(df[i].mean(),inplace = True)
    elif strategy == 'mode':
        for i in numericals:
            if df[i].isnull().sum() != 0:
                df[i].fillna(df[i].mode()[0],inplace = True)


# In[14]:


def missing(df):
    '''print a the names of the colums and the missing value in each column
    df: A pandas data frame, that you wish to sample from.
    '''
    for col in df.columns:
        print('Column {} is has {:.2f}% missing values'.format(col,df[col].isnull().sum()/df.shape[0]*100))


# In[4]:


def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = plt.figure(figsize = (9,5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000',           label = "Feature Weight")
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0',           label = "Cumulative Feature Weight")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()  


# In[18]:


def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))


# In[2]:


def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))


# In[6]:


def dendogram(df):
    '''plots a dendogram for the dataset to determine the correlated parameters. The correlation is calculated using ranked 
    features (spearman correlation) the column numbers should be even
    
    df: A pandas data frame, that you wish to sample from.'''
    
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=(16,10))
    dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
    plt.show()


# In[11]:


def feature_importance(m, df):
    '''return a dataframe of the random forest features importance sorted descendingly
    m: random forest model
    df: A pandas data frame, that you wish to sample from.'''
    
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


# In[8]:


def plot_feature_importance(feature_importance):
    '''plot for the feature importance'''
    return feature_importance.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[12]:


def fi_to_keep(fi,threshold=0.005):
    '''return the column names which has feature importance above a cerain threshold
    (use df = df[fi_to_keep] or create a copy in df_keep = df[fi_to_keep].copy())
    fi: a dataframe of feature importance (please use feature_importance function to generate the dataframe )
    threshold: the threshold for the feature importance'''
    return fi[fi.imp>threshold].cols


# In[ ]:


def cardinals(df,column,report = False):
    cum = 0
    cutoff = [len(df[column]),0]
    for index,i in enumerate(df[column].value_counts()):
        cum += (i/df.shape[0])
        if cum >= 0.8 and index < cutoff[0]:
            cutoff.pop()
            cutoff.pop()
            cutoff.append(index)
            cutoff.append(i/df.shape[0])
        if report == True:
            print('percentage is {:.4f} cum percentage is {:.2f}% curent index {}'.format(i/df.shape[0]*100,cum,index))
    return cutoff

def cardinals_list(df):
    cardinal_list=[]
    for i in is_categorical(df):
        cardinal = cardinals(df,i)
        if cardinal[1] != 0:
            cardinal_list.append([i,df[i].nunique(),cardinal])
            print ('Column {} has {} at index {} and threshold {}'.format(i,df[i].nunique(),cardinal[0],cardinal[1]))
    return cardinal_list



# In[1]:


def plot_pdp(df,model,feat, clusters=None, feat_name=None):
    '''Use a sample from the dataframe using get_sample()'''
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(model, df,df.columns,feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                        cluster=clusters is not None,
                        n_cluster_centers=clusters)

