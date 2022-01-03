import streamlit as st, random, ctypes, io, os, time, PIL
import pandas as pd, numpy as np, mediapipe as mp, cv2, pickle
from PIL import Image


# LOAD MODEL
with open('model_pred/model_28Dec.pkl', 'rb') as file:
    model = pickle.load(file)

# LOAD MEDIAPIPE
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils  
pose_img    = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.5)
pose_vid    = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.5)
pose_webcam = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5)


def predict_photo(image):
    # PROCESSING IMAGE FOR PREDICTING
    img_h, img_w, _         = image.shape
    results                 = pose_img.process(image)

    drawing = mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))

    img_lmk_values  = []
    if results.pose_landmarks:
        #11_LEFT_SHOULDER
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].z * img_w)
        #12_RIGHT_SHOULDER
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].z * img_w)
        #25_LEFT_ELBOW
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].z * img_w)
        #14_RIGHT_ELBOW
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].z * img_w)
        #15_LEFT_WRIST
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].z * img_w)
        #16_RIGHT_WRIST
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].z * img_w)

        #23_LEFT_HIP
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].z * img_w)
        #24_RIGHT_HIP
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].z * img_w)
        #25_LEFT_KNEE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].z * img_w)
        #26_RIGHT_KNEE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].z * img_w)
        #27_LEFT_ANKLE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].z * img_w)
        #28_RIGHT_ANKLE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].z * img_w)

        img_lmk_array   = list(np.array(img_lmk_values).flatten())
        global prediction, pred_proba_per, pred_proba
        X               = pd.DataFrame([img_lmk_array])
        prediction      = model.predict(X)[0]
        pred_proba      = model.predict_proba(X)[0]
        pred_proba_per  = round(pred_proba[np.argmax(pred_proba)],2)
        
    return drawing, prediction, pred_proba, pred_proba_per

def predict_vid(image):
    img_h, img_w, _         = image.shape
    results                 = pose_vid.process(image)

    drawing = mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
    img_lmk_values  = []
    if results.pose_landmarks:
        #11_LEFT_SHOULDER
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].z * img_w)
        #12_RIGHT_SHOULDER
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].z * img_w)
        #25_LEFT_ELBOW
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].z * img_w)
        #14_RIGHT_ELBOW
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].z * img_w)
        #15_LEFT_WRIST
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].z * img_w)
        #16_RIGHT_WRIST
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].z * img_w)

        #23_LEFT_HIP
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].z * img_w)
        #24_RIGHT_HIP
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].z * img_w)
        #25_LEFT_KNEE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].z * img_w)
        #26_RIGHT_KNEE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].z * img_w)
        #27_LEFT_ANKLE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].z * img_w)
        #28_RIGHT_ANKLE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].z * img_w)

        img_lmk_array   = list(np.array(img_lmk_values).flatten())
        global prediction, pred_proba, pred_proba_per
        X               = pd.DataFrame([img_lmk_array])
        prediction      = model.predict(X)[0]
        pred_proba      = model.predict_proba(X)[0]
        pred_proba_per  = round(pred_proba[np.argmax(pred_proba)],2)

    return drawing, prediction, pred_proba, pred_proba_per


def predict_webcam(image):
    img_h, img_w, _         = image.shape
    results                 = pose_webcam.process(image)

    drawing = mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
    img_lmk_values  = []
    if results.pose_landmarks:
        #11_LEFT_SHOULDER
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].z * img_w)
        #12_RIGHT_SHOULDER
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].z * img_w)
        #25_LEFT_ELBOW
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].z * img_w)
        #14_RIGHT_ELBOW
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].z * img_w)
        #15_LEFT_WRIST
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].z * img_w)
        #16_RIGHT_WRIST
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].z * img_w)

        #23_LEFT_HIP
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].z * img_w)
        #24_RIGHT_HIP
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].z * img_w)
        #25_LEFT_KNEE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].z * img_w)
        #26_RIGHT_KNEE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].z * img_w)
        #27_LEFT_ANKLE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].z * img_w)
        #28_RIGHT_ANKLE
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].x * img_w)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].y * img_h)
        img_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].z * img_w)

        img_lmk_array   = list(np.array(img_lmk_values).flatten())
        global prediction, pred_proba, pred_proba_per
        X               = pd.DataFrame([img_lmk_array])
        prediction      = model.predict(X)[0]
        pred_proba      = model.predict_proba(X)[0]
        pred_proba_per  = round(pred_proba[np.argmax(pred_proba)],2)

    return drawing, prediction, pred_proba, pred_proba_per


def show_image(image, pred_proba, pred_proba_per):
    percentage = str(pred_proba_per * 100) + '%'

    # DISPLAY ON IMAGE
    cv2.rectangle(image, (0,0), (image.shape[1], 20), (255, 255, 255), -1)
    if pred_proba[0] > 0.6:
        st.write('Prediction:', 'GOOD POSTURE', '(Probability: ' + str(percentage) + ')')
    elif pred_proba[0] < 0.45:
        st.write('Prediction:', 'POOR POSTURE', '(Probability: ' + str(percentage) + ')')
    else:
        st.write('Prediction:', 'DETECTING', '(Probability: ' + str(percentage) + ')')
    
def show_vid_web(image, pred_proba, pred_proba_per):
    percentage = str(pred_proba_per * 100) + '%'
    cv2.rectangle(image, (0,0), (image.shape[1], 80), (255, 255, 255), -1)
    if pred_proba[0] > 0.6:
        cv2.putText(image, 'GOOD POSTURE' + '  ' + str(pred_proba_per * 100) + '%', 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2, cv2.LINE_AA)
    elif pred_proba[0] < 0.45:
        cv2.putText(image, 'POOR POSTURE' + '  ' + str(pred_proba_per * 100) + '%', 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, 'DETECTING' + '  ' + str(pred_proba_per * 100) + '%', 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2, cv2.LINE_AA)