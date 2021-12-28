import streamlit as st, random, ctypes, io, os, time, PIL, tempfile
import pandas as pd, numpy as np, mediapipe as mp, cv2, pickle
from PIL import Image
from help_functions import get_lmk_array_img, get_lmk_array_vid, show_result



st.set_page_config(layout='centered', initial_sidebar_state='auto')


# LOAD PREDICTION MODEL & LOAD MEDIAPIPE
with open('model_pred/model_28Dec.pkl', 'rb') as file:
    model = pickle.load(file)
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils  


# BUILD STREAMLIT LAYOUT
menu = ['PicPred', 'VidPred', 'CamPred']
choice = st.sidebar.selectbox('Sitting Posture Recognition', menu)

with st.sidebar:
    st.text('') 
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('') 
    # Add in date & time in sidebar
    st.date_input('Today')
    st.time_input('Current time')


# DETECTION BY PHOTO UPLOAD
if choice == 'PicPred':
    st.title('Is Your Posture Right?'); st.text(""); st.write('Upload your picture here')

    with mp_pose.Pose(static_image_mode=True, 
                        min_detection_confidence=0.7, 
                            min_tracking_confidence=0.5) as pose:
        
        uploaded_photo = st.file_uploader('',['png', 'jpeg', 'jpg'])
        if uploaded_photo != None:
    
            # READING IMG
            image_np                = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
            image_np                = cv2.imdecode(image_np, 1)
            img_resized             = cv2.resize(image_np, (image_np.shape[1], image_np.shape[0]))

            # PROCESSING IMAGE FOR PREDICTING
            image                   = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            image.flags.writeable   = False
            results                 = pose.process(image)
            image.flags.writeable   = True
            image                   = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # DRAWING LANDMARKS & DISPLAY ON STREAMLIT 
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            st.image(image, channels='BGR')

            try:
                imp_lmk_array       = get_lmk_array_img(img_resized)
                X                   = pd.DataFrame([imp_lmk_array])
                prediction          = model.predict(X)[0]
                prediction_proba    = model.predict_proba(X)[0]
                pred_proba_per      = str(round(prediction_proba[np.argmax(prediction_proba)],2) * 100) + '%'
                # SHOW PREDICTION RESULT ON STREAMLIT
                result              = show_result(prediction_proba, pred_proba_per)
            except:
                pass


# DETECTION BY UPLOADING A VIDEO
elif choice == 'VidPred':
    st.text(""); st.write('Upload your video here')

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:

        uploaded_video  = st.file_uploader(' ', ['mp4', 'mov'])
        if uploaded_video is not None: 
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video = cv2.VideoCapture(tfile.name)

            stframe = st.empty()

            while video.isOpened(): 
                ret, frame = video.read()
                if not ret:
                    print("Can't get the frame")
                    break
                
                image                   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_h, frame_w, _     = frame.shape
                frame_copy              = frame.copy()
                image                   = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                image.flags.writeable   = False
                results                 = pose.process(image)
                image.flags.writeable   = True
                image                   = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # DRAWING LANDMARKS & DISPLAY ON STREAMLIT 
                mp_drawing.draw_landmarks(frame_copy, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
                # SHOWING REAL-TIME ON WEBCAM
                stframe.image(frame_copy,channels='BGR')

                try:
                    imp_lmk_array       = get_lmk_array_vid(image)
                    X                   = pd.DataFrame([imp_lmk_array])
                    prediction          = model.predict(X)[0]
                    prediction_proba    = model.predict_proba(X)[0]
                    pred_proba_per      = str(round(prediction_proba[np.argmax(prediction_proba)],2) * 100) + '%'

                    # SHOW PREDICTION RESULT ON STREAMLIT
                    result              = show_result(prediction_proba, pred_proba_per)
                    
                except:
                    st.write('DETECTING...')
      

# DETECTION WEBCAM (REAL-TIME)
elif choice == 'CamPred':

    with mp_pose.Pose(min_detection_confidence=0.7, 
                        min_tracking_confidence=0.5) as pose:

        cap = cv2.VideoCapture(0)  # device 0
        run = st.checkbox('Start running')

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        FRAME_WINDOW = st.image([])
        while run:
            cap.set(3, 640)
            cap.set(4, 480)

            ret, frame      = cap.read()
            if frame is not None:
                # READING IMG BY CV2
                frame                   = cv2.flip(frame, 1)
                frame_h, frame_w, _     = frame.shape
                frame_copy              = frame.copy()
                image                   = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                image.flags.writeable   = False
                results                 = pose.process(image)
                image.flags.writeable   = True
                image                   = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # DRAWING LANDMARKS & DISPLAY ON STREAMLIT 
                mp_drawing.draw_landmarks(frame_copy, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
                # SHOWING REAT-TIME ON WEBCAM
                FRAME_WINDOW.image(frame_copy,channels='BGR')

                try:
                    imp_lmk_array       = get_lmk_array_vid(image)
                    X                   = pd.DataFrame([imp_lmk_array])
                    prediction          = model.predict(X)[0]
                    prediction_proba    = model.predict_proba(X)[0]
                    pred_proba_per      = str(round(prediction_proba[np.argmax(prediction_proba)],2) * 100) + '%'

                    # SHOW PREDICTION RESULT ON STREAMLIT
                    result              = show_result(prediction_proba, pred_proba_per)
                    
                except:
                    st.write('DETECTING...')
              
    cap.release()

### THANK YOU! ###