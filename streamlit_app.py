import streamlit as st, random, ctypes, io, os, time, PIL, tempfile
import pandas as pd, numpy as np, mediapipe as mp, cv2, pickle
from PIL import Image
from help_functions import predict_vid, predict_webcam, predict_photo, show_image, show_vid_web

# SETUP STREAMLIT PAGE_CONFIG
st.set_page_config(layout='centered', initial_sidebar_state='auto')

# LOAD PREDICTION MODEL & LOAD MEDIAPIPE
with open('model_pred/model_28Dec.pkl', 'rb') as file:
    model = pickle.load(file)
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils  
pose_img    = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.5)
pose_vid    = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.5)
pose_webcam = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# BUILD STREAMLIT LAYOUT
homebox_title   = st.sidebar.write('HOME MENU')
welcome         = st.sidebar.checkbox('Hello!')
pic_pred        = st.sidebar.checkbox('PicPred')
vid_pred        = st.sidebar.checkbox('VidPred')
webcam_pred     = st.sidebar.checkbox('CamPred')

with st.sidebar:
    st.text(''); st.text(''); st.text(''); st.text(''); st.text(''); st.text('')
    st.date_input('Today'); st.time_input('Current time')

if welcome:
    st.image('media/streamlit/kidcover.png')
    
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
    st.audio('media/streamlit/We are the Children.mp3')
    st.write('Song: We are the worlds'); st.caption('Singer: Children group'); st.caption('Source: Internet')

# DETECTION BY PHOTO UPLOAD
if pic_pred:
    st.image('media/streamlit/dreamstime_88204858.png'); st.header('Do your kids sit correctly? Check it out!')

    uploaded_photo = st.file_uploader("Upload your kid's sitting picture!",['png', 'jpeg', 'jpg'])
    if uploaded_photo != None:

        # READING IMG
        image_np                = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
        image_np                = cv2.imdecode(image_np, 1)
        img_resized             = cv2.resize(image_np, (image_np.shape[1], image_np.shape[0]))

        # PROCESSING IMAGE FOR PREDICTING
        image                   = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        image.flags.writeable   = False
        results                 = pose_img.process(image)
        image.flags.writeable   = True
        image                   = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # GET RESULT
        drawing, prediction, pred_proba, pred_proba_per = predict_photo(image)

        # DISPLAY ON IMAGE ON STREAMLIT
        show_image(image, pred_proba, pred_proba_per)
        st.image(image, channels='BGR')
           

# DETECTION BY UPLOADING A VIDEO
if vid_pred:
    st.image('media/streamlit/cover3.png'); st.header("Let try with a video!")

    # UPLOAD VIDEO
    uploaded_video  = st.file_uploader('Upload your video here!', ['mp4', 'mov'])
    if uploaded_video is not None: 
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        frame_skip = 10     
        video.set(cv2. CAP_PROP_POS_FRAMES, frame_skip)
        while video.isOpened(): 
            ret, frame = video.read()
            if not ret:
                print("Can't get the frame")
                break
            
            image                   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_h, frame_w, _     = frame.shape
            image                   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable   = False
            results                 = pose_vid.process(image)
            image.flags.writeable   = True
            image                   = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            drawing, prediction, pred_proba, pred_proba_per = predict_vid(image)
            # DISPLAY ON STREAMLIT  
            show_vid_web(image, pred_proba, pred_proba_per)
            stframe.image(image)
    

# DETECTION WEBCAM (REAL-TIME)
if webcam_pred:
    st.header('Realtime detection'); st.caption('(Using local webcam)')

    cap = cv2.VideoCapture(0)  # device 0
    run = st.checkbox('Start running your webcam')

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    FRAME_WINDOW = st.image([])
    frame_skip = 10
    cap.set(cv2. CAP_PROP_POS_FRAMES, frame_skip)
    while run:
        cap.set(3, 640)
        cap.set(4, 480)

        ret, frame      = cap.read()
        if frame is not None:
            # READING IMG BY CV2
            frame                   = cv2.flip(frame, 1)
            frame_h, frame_w, _     = frame.shape
            image                   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            drawing, prediction, pred_proba, pred_proba_per = predict_webcam(image)
            # DISPLAY ON STREAMLIT
            show_vid_web(image, pred_proba, pred_proba_per)
            FRAME_WINDOW.image(image)

    cap.release()

### THANK YOU! ###