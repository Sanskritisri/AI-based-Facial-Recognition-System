import streamlit as st
import redis
import pandas as pd
import numpy as np
import cv2
from datetime import datetime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title='Predictions')
st.subheader('AI-based Facial Recognition System')

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

# Connect to Redis
hostname = 'redis-16088.c258.us-east-1-4.ec2.redns.redis-cloud.com'
portnumber = 16088
password = 'aBFqKVmQd8MRR2sfjzD4H7otEZS8jnqy'
r = redis.StrictRedis(
    host=hostname,
    port=portnumber,
    password=password
)

def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(
        lambda x: np.frombuffer(x, dtype=np.float32) if len(x) % 4 == 0 else None
    )
    retrive_series = retrive_series.dropna()  # Remove invalid entries
    index = retrive_series.index.map(lambda x: x.decode())
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role', 'Facial Features']
    retrive_df[['Name', 'Role']] = retrive_df['name_role'].str.split('@', expand=True)
    return retrive_df[['Name', 'Role', 'Facial Features']]

# Fetch data from Redis DB
with st.spinner('Retrieving data from Redis...'):
    redis_face_db = retrive_data(name='academy:register')
    st.dataframe(redis_face_db)
st.success("Data successfully retrieved.")

# Time
waitTime = 10

# Initialize face recognition
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model')
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

# Function to filter valid embeddings
def filter_valid_embeddings(dataframe, feature_column, expected_length=512):
    """Filter valid facial embeddings."""
    dataframe = dataframe.copy()
    dataframe[feature_column] = dataframe[feature_column].apply(
        lambda x: x if isinstance(x, np.ndarray) and x.shape[0] == expected_length else None
    )
    return dataframe.dropna(subset=[feature_column])

# Search algorithm for matching facial features
def ml_search_algorithm(dataframe, feature_column, test_vector, name_role=['Name', 'Role'], thresh=0.5):
    dataframe = filter_valid_embeddings(dataframe, feature_column)
    x_list = dataframe[feature_column].tolist()

    # Handle case where no valid embeddings exist
    if not x_list:
        return 'Unknown', 'Unknown'

    x = np.array(x_list)
    similar = cosine_similarity(x, test_vector.reshape(1, -1)).flatten()
    dataframe['cosine'] = similar

    # Filter data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if not data_filter.empty:
        argmax = data_filter['cosine'].idxmax()
        person_name = data_filter.at[argmax, name_role[0]]
        person_role = data_filter.at[argmax, name_role[1]]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role

class RealTimePred:
    def __init__(self):
        self.logs = {'name': [], 'role': [], 'current_time': []}

    def reset_dict(self):
        self.logs = {'name': [], 'role': [], 'current_time': []}

    def savelogs_redis(self):
        dataframe = pd.DataFrame(self.logs)
        dataframe.drop_duplicates('name', inplace=True)
        encoded_data = [
            f"{name}@{role}@{ctime}" for name, role, ctime in 
            zip(dataframe['name'], dataframe['role'], dataframe['current_time']) if name != 'Unknown'
        ]
        if encoded_data:
            r.lpush('attendance:logs', *encoded_data)
        self.reset_dict()

    def face_prediction(self, test_image, dataframe, feature_column, name_role=['Name', 'Role'], thresh=0.5):
        current_time = str(datetime.now())
        results = faceapp.get(test_image)
        test_copy = test_image.copy()

        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe, feature_column, embeddings, name_role, thresh)

            color = (0, 255, 0) if person_name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)
            cv2.putText(test_copy, person_name, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
            cv2.putText(test_copy, current_time, (x1, y2 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)

            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)

        return test_copy

def start_video_stream():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    video_placeholder = st.empty()
    realtimepred = RealTimePred()

    # Initialize setTime here
    setTime = time.time()  # Initialize setTime before it is used

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Failed to capture frame.")
            break

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            st.error("Captured frame is not a valid color image.")
            break

        pred_frame = realtimepred.face_prediction(frame, redis_face_db, 'Facial Features', ['Name', 'Role'])

        timenow = time.time()
        if timenow - setTime >= waitTime:
            realtimepred.savelogs_redis()
            setTime = time.time()  # Reset the timer

        frame_rgb = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        if st.button("Stop Stream", key="stop_button"):
            break

    cap.release()

if st.button("Start Stream", key="start_button"):
    start_video_stream()
