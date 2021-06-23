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

from Concrete_Strength_Predictor import return_preds,makepreds,optimize_recipe
import plotly.express as px

st.write("""
# Compressive Strength Predictor
Concrete is one of the most widely used materials in modern civil engineering efforts.
 Projects frequently require massive amounts of concrete of varying strengths, but evaluating the strength of concrete typically must wait until after the concrete has been made.
 """)
 
st.subheader("""Purpose
Using machine learning, this application predicts the **Compressive Strength in Megapascals(MPa)** based on user-specified concrete mixtures.
 This provides manufacturing and concrete suppliers with an accurate and dynamic evaluation of material strength, informing resource decisions ahead of time.
 Additionally, if the end-user has a desired compression strength requirement, he/she can input the constraint into the application and receive a breakdown of the optimal mixture components required.""")

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

st.subheader('Instructions\nTo view the predicted MPa of a mixture, use the sidebar marked **User Input Parameters in M3 Mixture**. This section will allow you to customize the parameters of a mixture. Chosen parameters will update the table marked **User Input Parameters** below, and return the predicted compressive strength of that mixture within the section marked **Predicted Compressive Strength of Mixture (+/- 3 MPa).**\n\nTo allow the app to develop a mixture for you based on required strength metrics, navigate to the section marked **Dynamically Develop a Mixture** below and follow additional instructions.')

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

preds = makepreds(df.iloc[0,0],df.iloc[0,1],df.iloc[0,2],df.iloc[0,3],df.iloc[0,4],df.iloc[0,5],df.iloc[0,6],df.iloc[0,7])
st.subheader('Predicted Compressive Strength of Mixture (+/- 3 MPa)')
st.write(str(round(preds,3)) + ' MPa')

st.subheader('Dynamically Develop a Mixture\nSelect a target strength value, # of iterations, grid space, and seed. From here, allow the app to develop a mixture for you that aims to be as close to the target strength requirement as possible! This feature will first prioritize searching for the closest solution that is greater than or equal to the target value. If this does not exist, then this feature will prioritize the closest solution that is less than the target value.')

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

if st.button('Develop Mixture', key='optimize_recipe'):
    pred,val1,val2,val3,val4,val5,val6,val7,val8,viz_df = optimize_recipe(target,iters,range_space,seed)
    st.write('**Predicted Compressive Strength of Developed Mixture: **' + str(round(pred,3)) + ' MPa')
    st.write("**Recipe of Developed Mixture:**")
    st.write('Cement: ' + str(round(val1,3)) + ' Kg, Slag: ' + str(round(val2,3)) + ' Kg, Fly Ash: ' + str(round(val3,3)) + 
    ' Kg, Water: ' + str(round(val4,3)) + ' Kg, Superplasticizer: ' + str(round(val5,3)) + ' Kg, Coarse Aggregate: ' + str(round(val6,3))
    + ' Kg, Fine Aggregate: ' + str(round(val7,3)) + ' Kg, Age: ' + str(round(val8,3)) + ' Days')
    viz_df = round(viz_df,3)
    newfeats = viz_df.iloc[:,:-1].copy()
    viz_scaler = pickle.load(open('viz_scaler.sav','rb'))
    viz_pca = pickle.load(open('viz_pca.sav','rb'))
    newfeats = viz_scaler.transform(newfeats)
    newfeats = pd.DataFrame(viz_pca.transform(newfeats))
    newfeats = pd.concat([newfeats,viz_df],axis=1)
    newfeats['Hover Name'] = newfeats['csMPa'].apply(lambda x:'Predicted Compressive Strength: ' + str(x) + ' MPa')
    newfeats.columns = ['Component 1','Component 2','cement','slag','flyash','water',
                    'superplasticizer','coarseaggregate','fineaggregate','age'
                    ,'csMPa','Hover Name']
    newfeats['label'] = newfeats.apply(lambda x:
                                  '<b>' + x['Hover Name'] + '</b><br><br><i>Mixture Recipe</i><br>'+
                                   'Cement: ' + str(x['cement']) +
                                   ' Kg<br>Slag: ' + str(x['slag']) +
                                   ' Kg<br>Fly Ash: ' + str(x['slag']) +
                                   ' Kg<br>Water: ' + str(x['slag']) +
                                   ' Kg<br>Super Plasticizer: ' + str(x['slag']) +
                                   ' Kg<br>Coarse Aggregate: ' + str(x['slag']) +
                                   ' Kg<br>Fine Aggregate: ' + str(x['slag']) +
                                   '<br>Age: ' + str(x['slag']) + ' Days',axis=1)
    fig = px.scatter(data_frame=newfeats,x='Component 1',y='Component 2',color = 'csMPa',hover_data=newfeats.iloc[:,2:-2],
                 hover_name='Hover Name',color_continuous_scale='RdYlGn')

    fig.update_layout(coloraxis_colorbar=dict(
        title="MPa",
    ),
        title='Mixture Development: All Results (Hover Over Data Points to View MPa and Mixture)',
        xaxis=dict(
            title='Component 1',
            gridcolor='white',
            gridwidth=2,
        ),
        yaxis=dict(
            title='Component 2',
            gridcolor='white',
            gridwidth=2))

    fig.update_traces(textfont_size = 2,
        hovertemplate = newfeats['label'],
        marker=dict(size=max(min(20,int(1000/len(newfeats))),5))
        )
    
    st.plotly_chart(fig)
    st.write('**Visualization Disclaimer**\n\nThe visualization above allows the user to graphically view the results of the mixture development and optimization process in order to provide added transparency on why the optimal mixture was chosen.'+
    ' Because there are 8 input variables, this implies that the initial data is 8D.'+
    ' The human brain can only visualize up to 3D, and this limitation often times can complicate visualizing high D results.' +
    ' Using an unsupervised machine learning technique called Principal Component Analysis (PCA), we can take high D data and map it onto lower dimensions while preserving as much information as possible.'+
    ' This app successfully utilizes PCA to map 8D data and predicted results onto 2D space.'+
    ' While this method makes our dimensions (Component 1, Component 2) difficult to interpret, it provides a great foundation for graphically representing our results.')


