import cv2, csv, mediapipe as mp, os, numpy as np, pathlib, glob, PIL, pickle, pandas as pd
from pathlib import Path
from PIL import UnidentifiedImageError, Image



# LOAD PREDICTION MODEL _ RANDOM FOREST CLASSIFIER
with open('model_pred/model_28Dec.pkl', 'rb') as file:
    model = pickle.load(file)


# LOAD MEDIAPIPE
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils  
pose        = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

def get_lmk_array(img):
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

def result_result(prediction_proba, pred_proba_per):
    if prediction_proba[0] >0.65:
            cv2.putText(frame_copy, 'GOOD', (380,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_copy, pred_proba_per, (200,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 117, 16), 2, cv2.LINE_AA)
    if prediction_proba[0] <=0.5:
        cv2.putText(frame_copy, 'POOR', (380,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_copy, pred_proba_per, (200,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 117, 16), 2, cv2.LINE_AA)
    if prediction_proba[0] >0.5 and prediction_proba[0] <=0.65:
        cv2.putText(frame_copy, 'UNKNOWN', (380,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_copy, pred_proba_per, (200,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 117, 16), 2, cv2.LINE_AA)


#SETUP WEBCAM
cap = cv2.VideoCapture(0)


if not cap.isOpened():                                      
    raise IOError("Cannot open webcam")

while cap.isOpened():
    cap.set(3, 640)
    cap.set(4, 480)

    frame_read  = 20
    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_read)
    
    ret, frame = cap.read()

    frame                   = cv2.flip(frame, 1)
    frame_h, frame_w, _     = frame.shape
    frame_copy              = frame.copy()
    image                   = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    image.flags.writeable   = False
    results                 = pose.process(image)
    image.flags.writeable   = True
    image                   = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(frame_copy, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))

    try:
        imp_lmk_array       = get_lmk_array(image)
        X                   = pd.DataFrame([imp_lmk_array])
        prediction          = model.predict(X)[0]
        prediction_proba    = model.predict_proba(X)[0]
        pred_proba_per      = str(round(prediction_proba[np.argmax(prediction_proba)],2) * 100) + '%'
        
        # BOX TO SHOW RESULT 
        cv2.rectangle(frame_copy, (0,0), (frame_w, 60), (255, 255, 255), -1)
        cv2.putText(frame_copy, 'Result', (310,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_copy, 'Prob', (150,40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # DISPLAY RESULT & PROBA 
        result_result(prediction_proba, pred_proba_per)
            
    except:
        pass

    cv2.imshow('Posture detection', frame_copy)

    c = cv2.waitKey(1)
    if c == 27:
        break


cap.release()
cv2.destroyAllWindows()
