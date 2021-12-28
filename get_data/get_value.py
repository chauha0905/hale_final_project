import cv2, csv, mediapipe as mp, os, numpy as np, pathlib, glob, PIL
from pathlib import Path
from PIL import UnidentifiedImageError, Image

############################################
label  = 'CORRECT_NOTAB'
mp_pose     = mp.solutions.pose
pose        = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.5)

#Function loop through dataset directory 
def get_lmk_array(img_path):
    img                 = cv2.imread(img_path)
    # img                 = cv2.flip(img,1)
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
    
        imp_lmk_array = list(np.array(imp_lmk_values).flatten())

    return imp_lmk_array

############################################ 
data_path = pathlib.Path('ds_26Dec/correct/CORRECT_NOTAB')

img_paths = list(data_path.glob('*'))
img_paths = [str(path) for path in img_paths]

for img_path in img_paths:
    if img_path.endswith('.jpg'):
        img = cv2.imread(img_path)
        
        imp_lmk_array = get_lmk_array(img_path)

        name          = os.path.basename(img_path)
        imp_lmk_array.insert(0, label)
        imp_lmk_array.insert(0, name)
            
        f = open('CORRECT_NOTAB.csv', mode='a', newline='')
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(imp_lmk_array)
        f.close()


    if not img_path.endswith('.jpg'):
        print(img_path)
        continue
