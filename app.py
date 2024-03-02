import streamlit as st
from streamlit_webrtc import webrtc_streamer,  RTCConfiguration, VideoProcessorBase, WebRtcMode

import av 
import numpy as np
import cv2
import pickle
import mediapipe as mp
import copy
from sklearn.preprocessing import StandardScaler
import pandas as pd


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
confidence = 0.5
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=confidence)


train_df = pd.read_csv('landmark_data/landmarks_train_v3.csv')
X_train = train_df.iloc[:, 1:].values
scaler = StandardScaler().fit(X_train)
with open('trained_models/random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="ISLT", page_icon="🤖")

st.title("Real Time Hand Gesture recognition and Live Hand Sign Translator")
st.text("Developed by Batch - 17")

predicted_gesture = ""

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)

        both_hand_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y))
                both_hand_landmarks.append(landmarks)
            
            if len(both_hand_landmarks) == 1:
                both_hand_landmarks.append([(0, 0)] * len(both_hand_landmarks[0]))
            values = list(np.array(both_hand_landmarks).flatten())
            values = scaler.transform([values])
            predicted = loaded_model.predict(values)
            cv2.rectangle(debug_image, (0,0), (160, 60), (245, 90, 16), -1)
        
            cv2.putText(debug_image, 'Predicted Gesture'
                        , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(debug_image, str(predicted[0])
                        , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(debug_image, format="bgr24")


webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
