import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("EduTech Course Price Predictor")

Institute = st.selectbox('Institute Name', df['Institute'].unique())
Course = st.selectbox('Course Name', df['Course'].unique())
Mode = st.selectbox('Online / Offline', ['Online' , 'Offline' , 'Both'])
Level = st.selectbox('Level', ['Biginner', 'Intermediate', 'Advance'])
Duration_Hours = st.selectbox('Duration Of Course (in Hours)',[10,15,20,30,40,50,60,90,100,120,130,140,150,160,180,200,210,250,280])
Trainer_Grade = st.selectbox('Trainers Grade', ['Low', 'Medium', 'High'])
Country = st.selectbox('Country', df['Country'].unique())
City = st.selectbox('City', df['City'].unique())
Certification_provided = st.selectbox('Certification_provided', ['Yes', 'No'])
Practice_Assignments = st.selectbox('Practice Assignments', ['Yes', 'No'])
Doubt_Sessions = st.selectbox('Doubt Sessions', ['Yes', 'No'])
Internship_Provided = st.selectbox('Internship Provided', ['Yes', 'No'])
Placement_Assistance = st.selectbox('Placement Assistance', ['Yes', 'No'])
Recorded_videos = st.selectbox('Provided Recorded videos', ['Yes', 'No'])

if st.button('Predict Price'):
    query = np.array(
        [Institute, Course, Mode, Level,Duration_Hours, Trainer_Grade, Country, City, Certification_provided,
         Practice_Assignments, Doubt_Sessions, Internship_Provided, Placement_Assistance, Recorded_videos])

    query = query.reshape(1, 14)
    st.title("The predicted price of this Course is  " + str(int(np.exp(pipe.predict(query)[0]))))

