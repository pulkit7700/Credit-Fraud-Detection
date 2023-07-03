import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from minisom import MiniSom
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import MinMaxScaler
from Model import som_, figure
import plotly.graph_objects as go
import pickle

st.title(':green[Credit Card Fraud Detection]')

with st.sidebar:
    st.image('https://media3.giphy.com/media/10YWA8gW28JbqM/giphy.gif?cid=ecf05e4705rmq4h3yyfbc4h6v1cgzx9h43wjfh5g7t4lfiry&ep=v1_gifs_search&rid=giphy.gif&ct=g')
    st.title("AutoInvestigate")
    choise = st.radio("Navigation", ['Upload', 'Profiling', 'Detecting', 'Show Information', 'Download'])
    st.info('AutoInvestigate is AI Deep Learning Model, that uses the power of Deep Learning to Detect anomoly and outputs data that might not be fiiting in the system, and labels it to be a Potential Fraud')

if os.path.exists('Data/Source_data.csv'):
    df2 = pd.read_csv('Data/Source_data.csv', index_col=None)

if choise == 'Upload':
    st.subheader(":blue[Upload]")
    file = st.file_uploader("Please Upload Your File", accept_multiple_files=False)
    if file:
        df2 = pd.read_csv(file,  index_col=None)
        df2.to_csv('Data/Source_data.csv', index=None)
        st.dataframe(df2)
    st.info('Please keep the Dependent Variable at the end of Data Variables for better performance')
if choise == 'Profiling':
    st.subheader(":blue[Exploratory Data Analysis]")
    profile_report = df2.profile_report()
    st_profile_report(profile_report)

if choise == 'Detecting':
    st.subheader('Detection of Anamolies')
    target = st.selectbox('Select Your Target', df2.columns)
    if st.button('Detect'):
        X = df2.iloc[:, :-1].values
        y = df2["{}".format(str(target))].values
        sc = MinMaxScaler(feature_range=(0, 1))
        X = sc.fit_transform(X)
        som = som_(df2)
        st.pyplot(figure(som, X, y))
        with open('som.pkl', 'wb') as pkl_file:
            pickle.dump(som, pkl_file)

if choise == 'Show Information':
        with open('som.pkl', 'rb') as pkl_file:
            som = pickle.load(pkl_file)
        st.write('Cordinates of First Box')
        t = st.number_input("Enter X Cordinate")
        z = st.number_input("Enter y Cordinate")
        st.write('Cordinates of Second Box')
        r = st.number_input("Enter second X Cordinate")
        k = st.number_input("Enter second y Cordinate")
        # names = pd.DataFrame(frauds, columns=df2.columns[0 : (len(df2.columns) - 1)])
        # st.dataframe(names)
        if st.button('Show Information'):
            X = df2.iloc[:, :-1].values
            sc = MinMaxScaler(feature_range=(0, 1))
            X = sc.fit_transform(X)        
            mappings = som.win_map(X)
            frauds = np.concatenate((mappings[(t, z)], mappings[(r, k)]), axis=0)
            frauds = sc.inverse_transform(frauds)
            file1 = pd.DataFrame(frauds, columns=['CustomerID', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9','A10', 'A11', 'A12', 'A13', 'A14'])
            file1.to_excel('Data/Frauds.xlsx', index=False)
            st.write(file1)

if choise == 'Download':
    with open('Data\Frauds.xlsx', 'rb') as f:
        st.download_button('Download the Frauds Data', f, 'Frauds.xlsx')
        
           
        
            
