import streamlit as st
import pandas as pd
from model import predict, ordinal_encoder
import numpy as np
from catboost import CatBoostClassifier

st.set_page_config (page_title = 'Income limit prediction',page_icon = '💹', layout = 'centered')

st.title(f'Income limit prediction')

employment_options = [' Children or Armed Forces', ' Not in labor force',
       ' PT for non-econ reasons usually FT', ' Full-time schedules',
       ' Unemployed full-time', ' PT for econ reasons usually PT',
       ' Unemployed part-time', ' PT for econ reasons usually FT']
gender_options = [' Male', ' Female']
industry_options = [' Not in universe or children', ' Business and repair services',
       ' Finance insurance and real estate',
       ' Manufacturing-durable goods', ' Wholesale trade',
       ' Hospital services', ' Manufacturing-nondurable goods',
       ' Public administration', ' Medical except hospital',
       ' Agriculture', ' Construction', ' Social services',
       ' Personal services except private HH', ' Education',
       ' Other professional services', ' Retail trade', ' Communications',
       ' Private household services', ' Transportation',
       ' Forestry and fisheries', ' Entertainment', ' Mining',
       ' Utilities and sanitary services', ' Armed Forces']
marital_options = [' Never married', ' Married-civilian spouse present', ' Divorced',
       ' Widowed', ' Separated', ' Married-spouse absent',
       ' Married-A F spouse present']
occ_options = [' Adm support including clerical', ' Sales',
       ' Executive admin and managerial',
       ' Precision production craft & repair',
       ' Handlers equip cleaners etc ', ' Protective services',
       ' Farming forestry and fishing',
       ' Transportation and material moving', ' Professional specialty',
       ' Other service', ' Private household services',
       ' Technicians and related support',
       ' Machine operators assemblers & inspectors', ' Armed Forces']
tax_options = [' Nonfiler', ' Joint one under 65 & one 65+', ' Single',
       ' Joint both under 65', ' Head of household', ' Joint both 65+']
edu_options = [' High school graduate', ' 12th grade no diploma', ' Children',
       ' Bachelors degree(BA AB BS)', ' 7th and 8th grade', ' 11th grade',
       ' 9th grade', ' Masters degree(MA MS MEng MEd MSW MBA)',
       ' 10th grade', ' Associates degree-academic program',
       ' 1st 2nd 3rd or 4th grade', ' Some college but no degree',
       ' Less than 1st grade', ' Associates degree-occup /vocational',
       ' Prof school degree (MD DDS DVM LLB JD)', ' 5th or 6th grade',
       ' Doctorate degree(PhD EdD)']

with st.form('Prediction form'):
    st.header('Enter the people specifications:')
    age = st.number_input (label='Age: ',min_value = 0, max_value = 90, step = 1)
    education = st.selectbox ('Select the type of education: ',options = edu_options)
    employment_commitment = st.selectbox('Select the type of employment: ',options = employment_options)
    gender = st.selectbox('Select gender: ',options= gender_options)
    industry_code_main = st.selectbox('Select the type of industry: ',options = industry_options)
    marital_status = st.selectbox('Select the marital status: ', options = marital_options)
    mig_year = st.number_input (label = 'Migration year: ',min_value = 94, max_value = 95, step = 1)
    occupation_code_main = st.selectbox('Select the type of occupation: ', options = occ_options)
    tax_status = st.selectbox ('Select the type of tax filer: ', options = tax_options)
    working_week_per_year = st.number_input (label='No. of working week per year: ',min_value = 1, max_value = 52, step = 1)
   
    
    submit_values = st.form_submit_button ('Predict')
    
if submit_values:
    education = ordinal_encoder(education,edu_options)
    employment_commitment = ordinal_encoder(employment_commitment,employment_options)
    gender = ordinal_encoder(gender,gender_options)
    industry_code_main = ordinal_encoder(industry_code_main,industry_options)
    marital_status = ordinal_encoder(marital_status, marital_options)
    occupation_code_main = ordinal_encoder(occupation_code_main,occ_options)
    tax_status = ordinal_encoder(tax_status,tax_options)
    
    data = np.array([age,education, employment_commitment,gender,industry_code_main,marital_status,
                     mig_year,occupation_code_main,tax_status, working_week_per_year]).reshape(1,-1)
    
    prediction = predict(data)
    
    if prediction[0] == 0:
        value = 'Person is below income limit'
    elif prediction[0] == 1:
        value = 'Person is above income limit'
    else:
        value = 'Error in prediction'
        
    st.header('Here is the prediction: ')
    st.success(f'{value}')
    st.balloons()
    
