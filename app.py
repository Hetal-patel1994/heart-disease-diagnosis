import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pgmpy.inference import VariableElimination

html_temp = """
    <div style="background-color:#f63366;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Disease Diagnosis App</h2>
    <p style="color:white;text-align:center;" >This is a <b>Streamlit</b> app used for prediction of chances of having a <b> Heart Disease</b> type.</p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

st.sidebar.header('About App')
st.sidebar.info("This web app helps you to find out whether you are at a risk of developing a heart disease.")
st.sidebar.info("Currently the app predicts data only for the given dataset. Please select the values based on the dataset to get desired output.")
st.sidebar.info("Enter the input and click on the 'Predict' button to check whether you have a healthy heart")

heart_dataset = pd.read_csv('heart_disease_uci.csv')
del heart_dataset['id']
del heart_dataset['dataset']
del heart_dataset['oldpeak']
del heart_dataset['slope']
del heart_dataset['ca']
del heart_dataset['thal']
heartDisease = heart_dataset.dropna()

max_age = heartDisease['age'].max()
min_age = heartDisease['age'].min()
max_trestbps = heartDisease['trestbps'].max()
min_trestbps = heartDisease['trestbps'].min()
max_chol = heartDisease['chol'].max()
min_chol = heartDisease['chol'].min()
max_thalch = heartDisease['thalch'].max()
min_thalch = heartDisease['thalch'].min()

result = heartDisease.apply(lambda x: x.sort_values().unique()[1], axis=0)
sec_min_trestbps = result['trestbps']
sec_min_chol = result['chol']


model_bayesian= pickle.load(open('heart_bayesian.pkl', 'rb'))
infer = VariableElimination(model_bayesian)

# Collects user input features into dataframe
def user_input_features():
    
        # following lines create boxes in which user can enter data required to make prediction
    age=st.slider('Age', min_age, max_age, min_age, 1)
    sex = st.radio("Select Gender: ", ('Male', 'Female')) 
    cp = st.selectbox('Chest Pain Type',("Typical angina","Atypical angina","Non-anginal pain","Asymptomatic")) 
    trestbps=st.slider('Resting blood pressure ', sec_min_trestbps, max_trestbps, sec_min_trestbps, 1.0)
    chol=st.slider('Serum cholestoral in mg/dl ', sec_min_chol, max_chol, sec_min_chol, 1.0)
    fbs=st.radio("Fasting Blood Sugar higher than 120 mg/dl", ['Yes','No'])
    restecg=st.selectbox('Resting Electrocardiographic Results',("Nothing to note","ST-T Wave abnormality","Possible or definite left ventricular hypertrophy"))
    thalch=st.slider('Maximum heart rate achieved ', min_thalch, max_thalch, min_thalch, 1.0)
    exang=st.radio('Exercise Induced Angina',["Yes","No"])  
    
    if cp=="Typical angina":
        cp='typical angina'
    elif cp=="Atypical angina":
        cp='atypical angina'
    elif cp=="Non-anginal pain":
        cp='non-anginal'
    elif cp=="Asymptomatic":
        cp='asymptomatic'   
         
    if exang=="Yes":
        exang=1
    elif exang=="No":
        exang=0
 
    if fbs=="Yes":
        fbs=1
    elif fbs=="No":
        fbs=0  
        
    if restecg=="Nothing to note":
        restecg='normal'
    elif restecg=="ST-T Wave abnormality":
        restecg='st-t abnormality'
    elif restecg=="Possible or definite left ventricular hypertrophy":
        restecg='lv hypertrophy'
         
    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalch':thalch,
            'exang':exang,
                }
    features = pd.DataFrame(data, index=[0])
    
    return features


st.subheader('Enter Input Through Slider')   
input_df = user_input_features()

st.subheader('User Input parameters')
st.dataframe(input_df)


# Apply model to make predictions
# Calculate the probability of getting heart disease
if st.button('PREDICT'):
  prediction = model_bayesian.predict(input_df)
  st.subheader('Final Predictions')
  st.write(f"#### Predicts Heart Disease Stage for Entered Data")
  prediction.columns=['Heart Disease Stage']
  st.dataframe(prediction)
  
  st.write(f"#### Predicts Heart Disease Probablity for Different Stages if no evidence is given")
  infer_prob = infer.query(['num'])
  prediction_prob = pd.DataFrame({'Heart Disease Stage': [0, 1, 2, 3, 4], 'Probablities': infer_prob.values.flatten()})
  st.dataframe(prediction_prob)
  
  st.write(f"#### Predicts Heart Disease Probablity for Different Stages for Entered Data")
  infer_prob_proper = infer.query(variables=['num'], evidence=input_df.iloc[0, :])
  prediction_prob_proper = pd.DataFrame({'Heart Disease Stage': [0, 1, 2, 3, 4], 'Probablities': infer_prob_proper.values.flatten()})
  st.dataframe(prediction_prob_proper)
  