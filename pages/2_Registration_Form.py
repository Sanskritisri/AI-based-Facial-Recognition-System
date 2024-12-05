import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time
import datetime
import redis
import os

st.set_page_config(page_title='Registration Form')
st.subheader('Registration Form')

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

# Step-1: Collect person name and role
person_name = st.text_input(label='Name', placeholder='First & Last Name')
role = st.selectbox(label='Select your Role', options=('Student', 'Teacher'))

# Initialize Redis connection
hostname = 'redis-16088.c258.us-east-1-4.ec2.redns.redis-cloud.com'
portnumber = 16088
password = 'aBFqKVmQd8MRR2sfjzD4H7otEZS8jnqy'
r = redis.StrictRedis(host=hostname, port=portnumber, password=password)

# Define RegistrationForm class
class RegistrationForm:
    def __init__(self):
        self.sample = 0  # Keeps track of the sample count

    def reset(self):
        self.sample = 0

    def get_embedding(self, frame):
        # Results from insightface
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        if len(results) > 0:
            for res in results:
                self.sample += 1  # Increment sample count
                x1, y1, x2, y2 = res['bbox'].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                text = f"samples = {self.sample}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                embeddings = res['embedding']
        else:
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame, embeddings

# Initialize face recognition
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model')
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

# Initialize RegistrationForm
registration_form = RegistrationForm()

# Optimized function to start webcam and process frames
def start_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    stframe = st.empty()  # Streamlit placeholder for displaying video frames
    last_time = time.time()  # To track frame rate

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture frame.")
            break

        # Process every nth frame to reduce lag (here, 10 FPS)
        if time.time() - last_time >= 0.1:
            reg_img, embedding = registration_form.get_embedding(frame)
            last_time = time.time()

            # Debugging: Check if embeddings are being detected
            if embedding is not None:
                st.write(f"Detected face and embedding sample {registration_form.sample}")
                # Save embeddings to a file
                file_path = 'C:\\Attendance System\\Notes\\4_attendance_app\\pages\\face_embeddings.txt'
                if not os.path.exists(os.path.dirname(file_path)):
                    os.makedirs(os.path.dirname(file_path))
                
                with open(file_path, mode='ab') as f:
                    np.savetxt(f, embedding)
                st.write(f"Embeddings saved successfully.")

            else:
                st.write("No face detected this frame.")

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(reg_img, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        # Exit loop if "Stop Camera" button is clicked
        if st.button("Stop Camera", key="stop_button"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start camera button
if st.button("Start Camera"):
    start_camera()

# Save the data in Redis database
def save_data_in_redis_db(name, role):
    if not name.strip():
        return 'name_false'

    # Check if the face_embeddings.txt file exists before proceeding
    file_path = 'C:\\Attendance System\\Notes\\4_attendance_app\\pages\\face_embeddings.txt'
    if not os.path.exists(file_path):
        return 'file_false'

    # Load the embeddings
    x_array = np.loadtxt(file_path)
    received_samples = int(x_array.size / 512)
    x_array = x_array.reshape(received_samples, 512)
    x_array = np.asarray(x_array)

    x_mean = x_array.mean(axis=0)
    x_mean = x_mean.astype(np.float32)
    x_mean_bytes = x_mean.tobytes()

    # Save data to Redis
    key = f'{name}@{role}'
    r.hset('academy:register', key=key, value=x_mean_bytes)  # Save name and role in Redis

    # Logging the attendance with name, role, and timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{name}@{role}@{current_time}"
    r.lpush('attendance:logs', log_entry)  # Store log entry in Redis

    # Cleanup
    os.remove(file_path)
    
    return True  # Return success

# Submit button
if st.button('Submit'):
    return_val = save_data_in_redis_db(person_name, role)
    if return_val is True:
        st.success(f"{person_name} registered successfully")
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces')
    elif return_val == 'file_false':
        st.error('face_embeddings.txt is not found. Please refresh the page and execute again.')
