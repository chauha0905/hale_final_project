import cv2, mediapipe as mp, os, numpy as np, pickle, pandas as pd, time, tempfile

# LOAD PREDICTION MODEL
with open('model_pred/model_28Dec.pkl', 'rb') as file:
    model = pickle.load(file)


# LOAD MEDIAPIPE
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils  
pose        = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# FUNCTION GET VALUES OF KEYPOINTS 
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
    
        # imp_lmk_array = list(np.array(imp_lmk_values).flatten())
        imp_lmk_array = np.array(imp_lmk_values).flatten()
        imp_lmk_array = list(imp_lmk_array)

    return imp_lmk_array


def result_result(prediction_proba, pred_proba_per):
    if prediction_proba[0] >0.6:
            cv2.putText(frame_copy, 'GOOD', (130,80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_copy, pred_proba_per, (420,80), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (245, 117, 16), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame_copy, 'POOR', (130,80), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_copy, pred_proba_per, (420,80), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (245, 117, 16), 2, cv2.LINE_AA)
    # if prediction_proba[0] >0.5 and prediction_proba[0] <=0.6:
    #     cv2.putText(frame_copy, 'UNKNOWN/DETECTING', (130,80), 
    #                 cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
    #     cv2.putText(frame_copy, pred_proba_per, (420,80), 
    #                 cv2.FONT_HERSHEY_DUPLEX, 1.5, (245, 117, 16), 2, cv2.LINE_AA)

# LOAD VIDEO
video_uploaded = cv2.VideoCapture('media/test_video/test_01.mov')

while video_uploaded.isOpened():
    # for frame in range(frame_count):
    # frame_skip  = 100
    # video_uploaded.set(cv2.CAP_PROP_POS_FRAMES, frame_skip)

    ret, frame = video_uploaded.read()
    image                   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _     = frame.shape
    frame_copy              = frame.copy()
    imgRGB                  = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    imgRGB.flags.writeable  = False
    results                 = pose.process(imgRGB)
    
    mp_drawing.draw_landmarks(frame_copy, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
    try:
        imp_lmk_array       = get_lmk_array(frame)
        X                   = pd.DataFrame([imp_lmk_array])
        prediction          = model.predict(X)[0]
        prediction_proba    = model.predict_proba(X)[0]
        pred_proba_per      = str(round(prediction_proba[np.argmax(prediction_proba)],2) * 100) + '%'

        
        # BOX TO SHOW RESULT 
        cv2.rectangle(frame_copy, (0,0), (frame_w, 120), (255, 255, 255), -1)
        # cv2.putText(frame_copy, 'Result', (310,40), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(frame_copy, 'Prob', (150,40), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # DISPLAY RESULT & PROBA 
        result_result(prediction_proba, pred_proba_per)

        cv2.imshow('POSTURE DECTECTION VIDEO', frame_copy)
       
    except:
        pass

    c = cv2.waitKey(1)
    if c == 27:
        break

    
video_uploaded.release()
cv2.destroyAllWindows()
