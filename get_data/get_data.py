import cv2, numpy as np, os, mediapipe as mp, io, csv

#LOAD MEDIAPOSE
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils 
pose        = mp_pose.Pose(static_image_mode=True, 
                            min_detection_confidence=0.5, 
                            min_tracking_confidence=0.5)

label = 'CORRECT_NOTAB'                   # TẠO LABEL CHO DATASET CSV

# FUNCTION GETTING 3-VALUES/12-KEYPOINTS
def get_lmk_array(img):             # BIẾN ĐẦU VÀO LÀ 1 HÌNH
    img                 = cv2.flip(img,1)
    img_h, img_w, _     = img.shape
    img_copy            = img.copy()
    imgRGB              = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    results             = pose.process(imgRGB)

    imp_lmk_values      = []
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
    return imp_lmk_array            # ARRAY CỦA 36-VALUES/12KEYPOINTS/IMG

# MỞ WEBCAM
cap         = cv2.VideoCapture(0)   
cap.set(3, 640)                 #SET FRAME SIZE FOR IMG_WIDTH/ WEIGHT
cap.set(4, 480)             

img_folder  = 'CORRECT_NOTAB'             # TẠO FOLDER LƯU IMAGE 
count       = 1                 # TẠO BIẾN ĐẾM CHO HÌNH 
try:                           
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
except OSError:
    print ('Error: Creating directory of data')

while cap.isOpened():
    ret, frame = cap.read()

    img_name = img_folder + '/Frame' + str(count) + '.jpg'
    print(img_name)
    cv2.imwrite(img_name, frame)    #LƯU HÌNH VÀO FOLDER

    frame                   = cv2.flip(frame, 1)
    frame_h, frame_w, _     = frame.shape
    frame_copy              = frame.copy()
    imageRGB                = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    results                 = pose.process(imageRGB)
    
    # DRAW LANDMARKS TRÊN HÌNH WEBCAM
    mp_drawing.draw_landmarks(frame_copy, 
                            results.pose_landmarks, 
                            mp_pose.POSE_CONNECTIONS)

    imp_lmk_array = get_lmk_array(frame) 

    imp_lmk_array.insert(0, label)
    imp_lmk_array.insert(0, 'Frame' + str(count) + '.jpg')

    # TẠO FILE CSV GHI DỮ LIỆU
    with open('INCORRECT_SHRIMP.csv', mode='a', newline='') as f: 
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(imp_lmk_array)
        
    count += 1
    cv2.imshow('GET DATA BY CAPTURING', frame_copy)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()