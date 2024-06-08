#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import streamlit as st
import numpy as np
from Prediction import predict
import warnings
warnings.filterwarnings("ignore")

st.title("Customer Churn Prediction")
st.markdown("Model to Predict the Customer")

st.header("Churn Features")
col1, col2, col3 = st.columns(3)
with col1 :
    tenure = st.slider("Tenure", 0,72,1)
with col2:
    MonthlyCharges = st.slider("Monthly Charges", 10.0, 200.0, 1.0)
with col3:
    TotalCharges = st.slider("Total Charges", 0.0, 10000.0,10.0)
    

if st.button("Predict MPG of Car"):
    result = predict(np.array([[tenure, MonthlyCharges, TotalCharges]]))
    st.text(result[0])



# In[ ]:




