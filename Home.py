import streamlit as st



st.set_page_config(page_title='Attendance System',layout='wide')

st.header('AI-based Facial Recognition System')

with st.spinner("Loading Models and Conneting to Redis db ..."):
    import face_rec
    
st.success('Model loaded sucesfully')
st.success('Redis db sucessfully connected')