import streamlit as st
import pandas as pd
import datetime
import redis
import numpy as np
from insightface.app import FaceAnalysis

st.set_page_config(page_title='Report', layout='wide')
st.subheader('Reporting')

from PIL import Image

import base64
from io import BytesIO

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load your PNG image (background)
image = Image.open("pages/gradient.png")  # Update the path if needed

# Display the background image as a full-screen background using CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('data:image/png;base64,{image_to_base64(image)}');
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialize face recognition
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model')
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

# Connect to Redis
hostname = 'redis-16088.c258.us-east-1-4.ec2.redns.redis-cloud.com'
portnumber = 16088
password = 'aBFqKVmQd8MRR2sfjzD4H7otEZS8jnqy'
r = redis.StrictRedis(host=hostname,
                     port=portnumber,
                     password=password)


class RealTimePred:
    def __init__(self):
        self.logs = {'name': [], 'role': [], 'current_time': []}

    def reset_dict(self):
        self.logs = {'name': [], 'role': [], 'current_time': []}

    def savelogs_redis(self):
        dataframe = pd.DataFrame(self.logs)
        dataframe.drop_duplicates('name', inplace=True)
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"  # Fixed the format
                encoded_data.append(concat_string)
        if len(encoded_data) > 0:
            r.lpush('attendance:logs', *encoded_data)

        self.reset_dict()  # Correctly reset the dictionary after saving logs

# Safe retrieval of facial feature embeddings from Redis
def safe_frombuffer(x):
    try:
        # Convert the data into a numpy array
        arr = np.frombuffer(x, dtype=np.float32)
        # Check if the length of the array is a multiple of 4 (size of np.float32)
        if len(arr) % 4 != 0:
            raise ValueError("Data size is not a multiple of 4 bytes")
        return arr
    except Exception as e:
        # Log the error or handle the corrupted data case
        print(f"Error processing data: {e}")
        return None

def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    # Apply safe_frombuffer to handle any data format issues
    retrive_series = retrive_series.apply(safe_frombuffer)
    
    # Filter out any None values (i.e., data that could not be processed)
    retrive_series = retrive_series.dropna()

    # Extract the name and role information
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))  # Decode from byte strings
    retrive_series.index = index

    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role', 'Facial Features']
    retrive_df[['Name', 'Role']] = retrive_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)

    return retrive_df[['Name', 'Role', 'Facial Features']]

# Retrieve the data from Redis
redis_face_db = retrive_data(name='academy:register')
dataframe = redis_face_db  # Using the dataframe retrieved from Redis
feature_column = 'Facial Features'  # The feature column name

# Function to load logs
def load_logs(name, end=-1):
    logs_list = r.lrange(name, start=0, end=end)  # Corrected: directly using the redis connection
    return logs_list

# Tabs to show the info
tab1, tab2 = st.tabs(['Registered Data', 'Logs'])

with tab1:
    if st.button('Refresh Data'):
        # Retrieve the data from Redis Database
        with st.spinner('Retrieving Data from Redis DB ...'):
            redis_face_db = retrive_data(name='academy:register')  # Call the function directly
            st.dataframe(redis_face_db[['Name', 'Role']])

with tab2:
    if st.button('Refresh Logs'):
        logs = load_logs(name='attendance:logs')
        st.write(logs)
