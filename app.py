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


with open('trained_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('trained_models/random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="ISLT", page_icon="🤖")

st.title("Real Time Hand Gesture recognition and Live Hand Sign Translator")
st.text("Developed by Batch - 17")

df = pd.read_csv("landmark_data/Gestures_sentences.csv")
my_dict = df.set_index('gesture_names')['sentence'].to_dict()
final_dict = {}
for key in my_dict:
    t = []
    words = key.split(',')
    for word in words:
        t.append(word)
    s = ' '.join(t)
    final_dict[s] = my_dict[key]


word_limit = 3 
def generate_caption(word, seq):
    res = ''
    if len(seq) < word_limit:
        seq.append(word)
        seq.append(word)
        s = ' '.join(seq)
        if s in final_dict:
            res = final_dict[s]

    elif len(seq) == word_limit:
        seq.pop(0)
        s = ' '.join(seq)
        if s in final_dict:
            res = final_dict[s]
        seq.append(word)
        s = ' '.join(seq)
        if s in final_dict:
            res = final_dict[s]   
    return res


class VideoProcessor(VideoProcessorBase):
    threshold_list = []
    threshold = 20
    seq = ['None']
    caption = ''
    prev_caption = ''
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
            self.threshold_list.append(predicted[0])
            if self.threshold_list.count(predicted[0]) >= self.threshold:
                # Add caption text
                if self.seq[-1] != predicted[0]:
                    self.caption = generate_caption(predicted[0], self.seq)
                if self.caption == '':
                    self.caption= self.prev_caption
                else:
                    self.prev_caption = self.caption
                self.threshold_list = []
            
        caption_size = cv2.getTextSize(self.caption, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        caption_x = int((debug_image.shape[1] - caption_size[0]) / 2)
        caption_y = debug_image.shape[0] - 10  # Adjust 10 for padding
        cv2.putText(debug_image, self.caption, (caption_x, caption_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(debug_image, format="bgr24")


webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
