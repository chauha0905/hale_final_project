import streamlit as st, random, ctypes, io, os, time, PIL
import pandas as pd, numpy as np, mediapipe as mp, cv2, pickle
from PIL import Image


# LOAD MODEL
with open('model_pred/model_28Dec.pkl', 'rb') as file:
    model = pickle.load(file)
# LOAD MEDIAPIPE
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils  


def get_lmk_array_img(img):
    pose = mp_pose.Pose(static_image_mode=True, 
                        min_detection_confidence=0.5, 
                        min_tracking_confidence=0.5)
    img                 = cv2.flip(img,1)
    img_h, img_w, _     = img.shape
    img_copy            = img.copy()
    imgRGB              = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    results             = pose.process(imgRGB)
    
    imp_lmk_values  = []
    if results.pose_landmarks:
        #11_LEFT_SHOULDER
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].z * img_w)
        #12_RIGHT_SHOULDER
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].z * img_w)
        #25_LEFT_ELBOW
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].z * img_w)
        #14_RIGHT_ELBOW
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].z * img_w)
        #15_LEFT_WRIST
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].z * img_w)
        #16_RIGHT_WRIST
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].z * img_w)

        #23_LEFT_HIP
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].z * img_w)
        #24_RIGHT_HIP
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].z * img_w)
        #25_LEFT_KNEE
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].z * img_w)
        #26_RIGHT_KNEE
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].z * img_w)
        #27_LEFT_ANKLE
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].z * img_w)
        #28_RIGHT_ANKLE
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].z * img_w)

        imp_lmk_array = np.array(imp_lmk_values).flatten()
        imp_lmk_array = list(imp_lmk_array)
    
    return imp_lmk_array


def get_lmk_array_vid(img):
    pose = mp_pose.Pose(static_image_mode=True, 
                        min_detection_confidence=0.7, 
                        min_tracking_confidence=0.5)
    img                 = cv2.flip(img,1)
    img_h, img_w, _     = img.shape
    img_copy            = img.copy()
    imgRGB              = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    results             = pose.process(imgRGB)

    imp_lmk_values  = []
    if results.pose_landmarks:
        #11_LEFT_SHOULDER
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(11).value].z * img_w)
        #12_RIGHT_SHOULDER
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(12).value].z * img_w)
        #25_LEFT_ELBOW
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(13).value].z * img_w)
        #14_RIGHT_ELBOW
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(14).value].z * img_w)
        #15_LEFT_WRIST
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(15).value].z * img_w)
        #16_RIGHT_WRIST
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(16).value].z * img_w)

        #23_LEFT_HIP
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(23).value].z * img_w)
        #24_RIGHT_HIP
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(24).value].z * img_w)
        #25_LEFT_KNEE
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(25).value].z * img_w)
        #26_RIGHT_KNEE
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(26).value].z * img_w)
        #27_LEFT_ANKLE
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(27).value].z * img_w)
        #28_RIGHT_ANKLE
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].x * img_w)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].y * img_h)
        imp_lmk_values.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(28).value].z * img_w)

        imp_lmk_array = np.array(imp_lmk_values).flatten()
        imp_lmk_array = list(imp_lmk_array)

    return imp_lmk_array



def show_result(prediction_proba, pred_proba_per):
    if prediction_proba[0] >0.6:
        st.write('Prediction:', '__GOOD POSTURE__', '(Probability: ' + str(pred_proba_per) + ')')
    if prediction_proba[0] <=0.5:
        st.write('Prediction:', '__POOR POSTURE__', '(Probability: ' + str(pred_proba_per) + ')')
    if prediction_proba[0] >0.5 and prediction_proba[0] <=0.6:
        st.write('Prediction:', '_UNKNOWN & DETECTING...__', '(Probability: ' + str(pred_proba_per) + ')')
