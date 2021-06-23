import streamlit as st
import pandas as pd
import numpy as np
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

import pickle

st.write("""
# Compressive Strength Predictor 
This app predicts the **Compressive Strength** of concrete mixtures, while also being able to dynamically develop mixtures based off required strength metrics.
""")

st.sidebar.header('User Input Parameters in M3 Mixture')

def user_input_features(val1=321.0,val2=179.7,val3=100.05,val4=184.4,val5=16.1,val6=973.0,val7=793.3,val8=183.0):
    cement = st.sidebar.slider('Cement (Kg)', 102.0, 540.0, val1)
    slag = st.sidebar.slider('Slag (Kg)', 0.0, 359.4, val2)
    flyash = st.sidebar.slider('Fly Ash (Kg)', 0.0, 200.1, val3)
    water = st.sidebar.slider('Water (Kg)', 121.8, 247.0, val4)
    superplasticizer = st.sidebar.slider('Superplasticizer (Kg)', 0.0, 32.2, val5)
    coarseagg = st.sidebar.slider('Coarse Aggregate (Kg)', 801.0, 1145.0, val6)
    fineagg = st.sidebar.slider('Fine Aggregate (Kg)', 594.0, 992.6, val7)
    age = st.sidebar.slider('Age (Days)', 1.0, 365.0, val8)


    data = {'Cement': cement,
            'Slag': slag,
            'Fly Ash': flyash,
            'Water': water,
            'Superplasticizer': superplasticizer,
            'Coarse Aggregate': coarseagg,
            'Fine Aggregate': fineagg,
            'Age': age}
    features = pd.DataFrame(data, index=[0])
    return features

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

def makepreds(cement,slag,flyash,water,superplasticizer,coarseaggregate,fineaggregate,age):
    
    feature_engineering = pickle.load(open('feature_engineering.sav', 'rb'))
    layers = []
    layer_cols = []
    for num in range(1,9):
        layers.append(pickle.load(open('l' + str(num) + '_models.sav','rb')))
        layer_cols.append(pickle.load(open('l' + str(num) + '_cols.sav','rb')))
    
    preds = return_preds(cement,slag,flyash,water,superplasticizer,coarseaggregate,fineaggregate,age,feature_engineering,layers,layer_cols)
    return preds

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

preds = makepreds(df.iloc[0,0],df.iloc[0,1],df.iloc[0,2],df.iloc[0,3],df.iloc[0,4],df.iloc[0,5],df.iloc[0,6],df.iloc[0,7])
st.subheader('Predicted Compressive Strength of Mixture (+/- 3 MPa)')
st.write(str(round(preds,3)) + ' MPa')

st.subheader('Dynamically Develop a Mixture\nSelect a target strength value, # of iterations, grid space, and seed. From here, allow the app to develop a mixture for you that aims to be as close to the target strength requirement as possible!')

target = st.selectbox('Target Compressive Strength (MPa):',
(10,15,20,25,30,35,40,45,50,55,60,65,70,75,80))

iters = st.selectbox('Number of Iterations: The higher the # the increased chance you give the app to find an optimal solution.',
(10,50,100,200,300,400,500,1000))

range_space = st.selectbox('Grid Space: The higher the # the greater the amount of possibilities the app will account for with respect to each input parameter.',
(10,20,30,40,50,60,70,80,90,100))

seed = st.selectbox('Seed: Because this feature utilizes Monte Carlo style methods, specifying a seed value ensures reproducable results.',
(10,20,30,40,50,60,70,80,90,100))


st.write('**You selected:**\n\nTarget Compressive Strength (MPa): ' + str(target) + ', Number of Iterations: ' + 
str(iters) + ', Grid Space: ' + str(range_space) + ', Seed: ' + str(seed))


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
    
    current_pred = makepreds(cem_range[0],slag_range[0],flyash_range[0],water_range[0],superplasticizer_range[0],coarseaggregate_range[0],fineaggregate_range[0],age_range[0])

    s = [[0,0,0,0,0,0,0,0]]
    s_preds = [current_pred]
    
    np.random.seed(seed)
    for num in range(iters):
        s.append(list(np.random.randint(0,range_space,8)))
        s_preds.append(return_preds(cem_range[s[-1][0]],slag_range[s[-1][1]],flyash_range[s[-1][2]],water_range[s[-1][3]],
                                   superplasticizer_range[s[-1][4]],coarseaggregate_range[s[-1][5]],
                                   fineaggregate_range[s[-1][6]],age_range[s[-1][7]],feature_engineering,layers,layer_cols))

        
    s_dist = np.array([x-target for x in s_preds])
    
    if len(s_dist[s_dist>=0]) == 0:
        best_id = s_dist.argmax()
    else:
        best_id = s_dist[s_dist>=0].min()
        best_id = list(s_dist).index(best_id)
    
    return s_preds[best_id],cem_range[s[best_id][0]],slag_range[s[best_id][1]],flyash_range[s[best_id][2]],water_range[s[best_id][3]],superplasticizer_range[s[best_id][4]],coarseaggregate_range[s[best_id][5]],fineaggregate_range[s[best_id][6]],age_range[s[best_id][7]]

if st.button('Develop Mixture', key='optimize_recipe'):
    pred,val1,val2,val3,val4,val5,val6,val7,val8 = optimize_recipe(target,iters,range_space,seed)
    st.write('**Compressive Strength of Developed Mixture: **' + str(round(pred,3)) + ' MPa')
    st.write("**Recipe of Developed Mixture:**")
    st.write('Cement: ' + str(round(val1,3)) + ', Slag: ' + str(round(val2,3)) + ', Fly Ash: ' + str(round(val3,3)) + 
    ', Water: ' + str(round(val4,3)) + ', Superplasticizer: ' + str(round(val5,3)) + ', Coarse Aggregate: ' + str(round(val6,3))
    + ', Fine Aggregate: ' + str(round(val7,3)) + ', Age: ' + str(round(val8,3)))