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
homebox_title   = st.sidebar.write('HOME MENU')
welcome         = st.sidebar.checkbox('Hello!')
pic_pred        = st.sidebar.checkbox('PicPred')
vid_pred        = st.sidebar.checkbox('VidPred')
webcam_pred     = st.sidebar.checkbox('CamPred')

with st.sidebar:
    st.text(''); st.text(''); st.text(''); st.text(''); st.text(''); st.text('')
    # Add in date & time in sidebar
    st.date_input('Today'); st.time_input('Current time')

if welcome:
    st.image('media_streamlit/kidcover.png')
    
    col1, col2, col3 = st.columns([1,4,1])
    with col1:
        st.empty()
    with col2:
        st.subheader('Children are treasure!')
    with col3:
        st.empty()
    col4, col5, col6 = st.columns([1,2,1])
    with col4:
        st.empty()
    with col5:
        st.write('Care - Protect - Support')
    with col6:
        st.empty()
    st.audio('media_streamlit/We are the Children.mp3')
    st.write('Song: We are the worlds'); st.caption('Singer: Children group'); st.caption('Source: Internet')

    
# DETECTION BY PHOTO UPLOAD
if pic_pred:
    st.image('media_streamlit/dreamstime_88204858.png'); st.header('Do your kids sit correctly? Check it out!')

    with mp_pose.Pose(static_image_mode=True, 
                        min_detection_confidence=0.7, 
                            min_tracking_confidence=0.5) as pose:
        
        uploaded_photo = st.file_uploader("Upload your kid's sitting picture!",['png', 'jpeg', 'jpg'])
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
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
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
            
            # SAVE IMAGE UPLOADED BY USER ON STREAMLIT (for improving model in the future)
            user_folder = 'user_photo'
            user_count = 1
            try:                           
                if not os.path.exists(user_folder):
                    os.makedirs(user_folder)
            except:
                pass
            img_saved_name  = user_folder + 'user' + str(user_count) +'.jpg'
            img_saved       = cv2.imwrite(img_saved_name, image)



# DETECTION BY UPLOADING A VIDEO
if vid_pred:
    st.image('media_streamlit/cover3.png')
    st.header("Let try with a video!")

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:

        uploaded_video  = st.file_uploader('Upload your video here!', ['mp4', 'mov'])
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
if webcam_pred:
    st.header('Realtime detection'); st.caption('(Using local webcam)')

    with mp_pose.Pose(min_detection_confidence=0.7, 
                        min_tracking_confidence=0.5) as pose:

        cap = cv2.VideoCapture(0)  # device 0
        run = st.checkbox('Start running your webcam')

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