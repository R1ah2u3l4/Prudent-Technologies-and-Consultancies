#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import numpy as np
import pickle


# In[2]:


col1,col2,col3,col4,col5,col6,col7 = st.columns(7)


# In[8]:


import warnings    ## To surpass the warnings from the data
warnings.filterwarnings("ignore")


# In[9]:


# Define Streamlit app layout and widgets
st.title('Customer Churn Prediction')


# In[10]:


with col1:
    ##Tenure, Monthlycharges, Total_charges
    tenure = st.slider('tenure', min_value = 1, max_value =72, value=24)
    MonthlyCharges = st.slider('MonthlyCharges', min_value=18.0, max_value=119.0, value=70.0, step=0.25)
    TotalCharges = st.number_input('TotalCharges', min_value=0.0, max_value=8465.0, value=3547.0, step=1.0)  


# In[11]:


with col2:
    #Gender
    gender_display = ('gender_Female', 'gender_Male')
    gender_options = list(range(len(gender_display)))
    gender = st.selectbox('gender', gender_options, format_func=lambda x:gender_display[x])
    
    #Senior Citizen
    senior_citizen_display=('SeniorCitizen_0', 'SeniorCitizen_1')
    senior_citizen_options=list(range(len(senior_citizen_display)))
    SeniorCitizen=st.selectbox('SeniorCitizen',senior_citizen_options,format_func=lambda x:senior_citizen_display[x])
    
    #Partner display
    partner_display=('Partner_No', 'Partner_Yes')
    partner_options=list(range(len(partner_display)))
    Partner=st.selectbox('Partner',partner_options,format_func=lambda x:partner_display[x])


# In[14]:


with col3:
    #Dependents
    dependents_display=('Dependents_No', 'Dependents_Yes')
    dependents_options=list(range(len(dependents_display)))
    Dependents=st.selectbox('Dependents',dependents_options,format_func=lambda x:dependents_display[x])
    
    #Phoneservice
    phoneservice_display=('PhoneService_No', 'PhoneService_Yes')
    phoneservice_options=list(range(len(phoneservice_display)))
    PhoneService=st.selectbox('PhoneService',phoneservice_options,format_func=lambda x:phoneservice_display[x])
    
    #Multiple Lines
    multiplelines_display=('MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes')
    multiplelines_options=list(range(len(multiplelines_display)))
    MultipleLines=st.selectbox('MultipleLines',multiplelines_options,format_func=lambda x:multiplelines_display[x])


# In[17]:


with col4:
    #Internet Service
    internetservice_display=('InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No')
    internetservice_options=list(range(len(internetservice_display)))
    InternetService=st.selectbox('InternetService',internetservice_options,format_func=lambda x:internetservice_display[x])
    
    #Online security
    onlinesecurity_display=('OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes')
    onlinesecurity_options=list(range(len(onlinesecurity_display)))
    OnlineSecurity=st.selectbox('OnlineSecurity',onlinesecurity_options,format_func=lambda x:onlinesecurity_display[x])
    
    #online backup
    onlinebackup_display=('OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes')
    onlinebackup_options=list(range(len(onlinebackup_display)))
    OnlineBackup=st.selectbox('OnlineBackup',onlinebackup_options,format_func=lambda x:onlinebackup_display[x])


# In[18]:


with col5:
    # Device Protection
     deviceprotection_display=('DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes')
     deviceprotection_options=list(range(len(deviceprotection_display)))
     DeviceProtection=st.selectbox('DeviceProtection',deviceprotection_options,format_func=lambda x:deviceprotection_display[x])
    
    # Tech support 
     techsupport_display=('TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes')
     techsupport_options=list(range(len(techsupport_display)))
     TechSupport=st.selectbox('TechSupport',techsupport_options,format_func=lambda x:techsupport_display[x])
    
    # streaming Tv
     streamingtv_display=('StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes')
     streamingtv_options=list(range(len(streamingtv_display)))
     StreamingTV=st.selectbox('StreamingTV',streamingtv_options,format_func=lambda x:streamingtv_display[x])


# In[20]:


with col6:
    # Streaming movies
     streamingmovies_display=('StreamingMovies_No', 'StreamingMovies_No internet service',  'StreamingMovies_Yes')
     streamingmovies_options=list(range(len(streamingmovies_display)))
     StreamingMovies=st.selectbox('StreamingMovies',streamingmovies_options,format_func=lambda x:streamingmovies_display[x])

    # Payment 
     paymentmethod_display=('PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check')
     paymentmethod_options=list(range(len(paymentmethod_display)))
     PaymentMethod=st.selectbox('PaymentMethod',paymentmethod_options,format_func=lambda x:paymentmethod_display[x])

    # Paperless billing
     paperless_billing_display=('PaperlessBilling_No', 'PaperlessBilling_Yes')
     paperless_billing_options=list(range(len(paperless_billing_display)))
     PaperlessBilling=st.selectbox('PaperlessBilling',paperless_billing_options,format_func=lambda x:paperless_billing_display[x])


# In[21]:


with col7:
    # contract 
     contract_display=('Contract_Month-to-month', 'Contract_One year', 'Contract_Two year')
     contract_options=list(range(len(contract_display)))
     Contract=st.selectbox('Contract',contract_options,format_func=lambda x:contract_display[x])

    # Churn
     churn_display=('Churn_No', 'Churn_Yes')
     churn_options=list(range(len(churn_display)))
     Churn=st.selectbox('Churn',churn_options,format_func=lambda x:churn_display[x])


# In[22]:


import os
file_path = os.path.join("C:\\Users\\hp\\Desktop\\1.2 Customer churn", "gbcmodel.pkl")


# In[23]:


model = pickle.load(open(file_path, 'rb'))


# In[24]:


## Create a dictionary 
my_dict = {
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'Churn': Churn,
}


# In[25]:


df = pd.DataFrame([my_dict])
df = pd.get_dummies(df).reindex(columns=df.columns, fill_value=0)


# In[26]:


if st.button("Predict"):
    result = model.predict(df)
    probability =model.predict_proba(df)
    st.text(result[0])
    st.text(probability[0])

