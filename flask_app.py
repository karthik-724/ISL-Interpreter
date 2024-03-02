from flask import Flask, render_template, Response
import numpy as np
import cv2
import pickle
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import pandas as pd


app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
confidence = 0.5
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=confidence)


train_df = pd.read_csv('landmark_data/landmarks_train_v3.csv')
X_train = train_df.iloc[:, 1:].values
scaler = StandardScaler().fit(X_train)
with open('trained_models/random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)


predicted = ['None']



def generate_frames():
    cap = cv2.VideoCapture(0)  
    while True:
        ret, image = cap.read()

        if not ret:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        
        #converting the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        both_hand_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    # Extract x, y coordinates (relative to image dimensions)
                    x = landmark.x
                    y = landmark.y
                    # Append coordinates to the list
                    landmarks.append((x, y))
                both_hand_landmarks.append(landmarks)
            
            if len(both_hand_landmarks) == 1:
                both_hand_landmarks.append([(0, 0)] * len(both_hand_landmarks[0]))
            values = list(np.array(both_hand_landmarks).flatten())
            values = scaler.transform([values])
            predicted = loaded_model.predict(values)
            cv2.rectangle(image, (0,0), (160, 60), (245, 90, 16), -1)
            # Displaying Class
            cv2.putText(image, 'Predicted Gesture'
                        , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(predicted[0])
                        , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n' )

# Route to render the webpage
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream the camera frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)