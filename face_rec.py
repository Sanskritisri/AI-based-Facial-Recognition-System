from insightface.app import FaceAnalysis
import pandas as pd
import numpy as np
import cv2
import redis
import time
from datetime import datetime

from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import cosine_similarity

#connect to redis
hostname = 'redis-16088.c258.us-east-1-4.ec2.redns.redis-cloud.com'
portnumber = 16088
password = 'aBFqKVmQd8MRR2sfjzD4H7otEZS8jnqy'
r = redis.StrictRedis(host=hostname,
                     port=portnumber,
                     password=password)

# Extract data from db academy:register
def retrive_data(name):
    #name = 'academy:register'
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x:np.frombuffer(x,dtype = np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role','Facial Features']
    retrive_df[['Name','Role']] = retrive_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['Name','Role','Facial Features']]

#face analysis
faceapp = FaceAnalysis(name='buffalo_sc',
                    root='insightface_model')
faceapp.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5)


def ml_search_algorithm(dataframe, feature_column, test_vector, name_role=['Name','Role'], thresh=0.5):
    
    x_list = dataframe[feature_column].tolist()
    x = np.asarray(x_list)
    
    similar = cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    #FILTER DATA
    data_filter = dataframe.query(f'cosine>={thresh}')
    if len(data_filter)>0:
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name = data_filter.loc[argmax, name_role[0]]
        person_role = data_filter.loc[argmax, name_role[1]]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name,person_role
    

#multi face detection
def face_prediction(test_image,dataframe, feature_column, name_role=['Name','Role'], thresh=0.5):
    # date time
    current_time = str(datetime.now())
    
    results = faceapp.get(test_image)
    test_copy = test_image.copy()
    
    for res in results:
        x1,y1,x2,y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
        person_name, person_role = ml_search_algorithm(dataframe,
                                                       feature_column,
                                                       test_vector=embeddings,
                                                      name_role=name_role,
                                                      thresh=thresh)
        #unknown in red
        if person_name == 'Unknown':
            color =(0,0,255)
        else:
            color =(0,255,0)
            
        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
        text_gen = person_name
        cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)
        cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.5,color,2)
    return test_copy
