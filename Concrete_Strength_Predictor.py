#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

import pickle


# In[2]:


def return_preds(cement,slag,flyash,water,superplasticizer,coarseaggregate,fineaggregate,age,feature_engineering,layers,layer_cols):

    newdf = pd.concat([feature_engineering[0],pd.DataFrame([cement,slag,flyash,water,superplasticizer,coarseaggregate,fineaggregate,age],index=feature_engineering[0].columns).T],axis=0)

    for a,b in feature_engineering[1]:
        if newdf[a].min() != 0.0 and newdf[b].min() != 0:
            newdf[str(a) + '/' + str(b)] = newdf[a]/newdf[b]
        elif newdf[a].min() == 0 and newdf[b].min() != 0:
            newdf[str(a) + '/' + str(b)] = newdf[a]/newdf[b]
        elif newdf[a].min() != 0 and newdf[b].min() == 0:
            newdf[str(b) + '/' + str(a)] = newdf[b]/newdf[a]

    for cols in newdf.columns:
        if newdf[cols].min() != 0:
            newdf['1/('+str(cols) + ')'] = 1/newdf[cols]
            newdf['ln ' + str(cols)] = np.log(newdf[cols])

    newdf = newdf.iloc[-1]
    newdf = pd.DataFrame(feature_engineering[3].transform(feature_engineering[2].transform(pd.DataFrame(newdf).T)))
    
    count = 1
    for num in range(8):
        preds = []
        if num != 7:
            for num1 in range(6):
                preds.append(layers[num][num1].predict(newdf[layer_cols[num][num1]]))

            newdf = pd.DataFrame(preds).T
            newdf.columns = np.arange(count,count+6)
            count = count+6

        else:
            preds = layers[7].predict(newdf[layer_cols[7]])
            preds = preds[0]
            
    return preds
            
    


# In[3]:


def makepreds(cement,slag,flyash,water,superplasticizer,coarseaggregate,fineaggregate,age):
    
    feature_engineering = pickle.load(open('feature_engineering.sav', 'rb'))
    layers = []
    layer_cols = []
    for num in range(1,9):
        layers.append(pickle.load(open('l' + str(num) + '_models.sav','rb')))
        layer_cols.append(pickle.load(open('l' + str(num) + '_cols.sav','rb')))
    
    preds = return_preds(cement,slag,flyash,water,superplasticizer,coarseaggregate,fineaggregate,age,feature_engineering,layers,layer_cols)
    return preds


# In[14]:


def optimize_recipe(target = 70,iters = 50,range_space = 30,seed = 50):
    feature_engineering = pickle.load(open('feature_engineering.sav', 'rb'))
    layers = []
    layer_cols = []
    for num in range(1,9):
        layers.append(pickle.load(open('l' + str(num) + '_models.sav','rb')))
        layer_cols.append(pickle.load(open('l' + str(num) + '_cols.sav','rb')))
    
    cem_range = np.linspace(102,540,range_space)
    slag_range = np.linspace(0,359.4,range_space)
    flyash_range = np.linspace(0,200.1,range_space)
    water_range = np.linspace(121.8,247,range_space)
    superplasticizer_range = np.linspace(0,32.2,range_space)
    coarseaggregate_range = np.linspace(801,1145,range_space)
    fineaggregate_range = np.linspace(594,992.6,range_space)
    age_range = np.linspace(1,365,range_space)
    
    s = []
    s_preds = []
    s_viz = []
    
    np.random.seed(seed)
    for num in range(iters):
        s.append(list(np.random.randint(0,range_space,8)))
        s_preds.append(return_preds(cem_range[s[-1][0]],slag_range[s[-1][1]],flyash_range[s[-1][2]],water_range[s[-1][3]],
                                   superplasticizer_range[s[-1][4]],coarseaggregate_range[s[-1][5]],
                                   fineaggregate_range[s[-1][6]],age_range[s[-1][7]],feature_engineering,layers,layer_cols))
        s_viz.append([cem_range[s[-1][0]],slag_range[s[-1][1]],flyash_range[s[-1][2]],water_range[s[-1][3]],
                                   superplasticizer_range[s[-1][4]],coarseaggregate_range[s[-1][5]],
                                   fineaggregate_range[s[-1][6]],age_range[s[-1][7]]])

        
    s_dist = np.array([x-target for x in s_preds])
    s_viz = pd.DataFrame(s_viz)
    s_viz.columns = ['cement','slag','flyash','water',
                    'superplasticizer','coarseaggregate','fineaggregate','age']
    s_viz['csMPa'] = s_preds
    
    if len(s_dist[s_dist>=0]) == 0:
        best_id = s_dist.argmax()
    else:
        best_id = s_dist[s_dist>=0].min()
        best_id = list(s_dist).index(best_id)
    
    return s_preds[best_id],cem_range[s[best_id][0]],slag_range[s[best_id][1]],flyash_range[s[best_id][2]],water_range[s[best_id][3]],superplasticizer_range[s[best_id][4]],coarseaggregate_range[s[best_id][5]],fineaggregate_range[s[best_id][6]],age_range[s[best_id][7]],s_viz


# In[17]:


optimize_recipe(75,50,30,30)


# In[ ]:




