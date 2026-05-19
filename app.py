# app.py
"""
Streamlit app for Bharatnatyam Mudra Recognition.

To run this application:
1. Ensure you have the trained model artifacts in the 'model_artifacts' directory.
   If not, run 'model_train.py' first.
2. Ensure you have the reference mudra images in the 'mudra_images' directory.
3. Run the command: streamlit run app.py
"""

import os
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import pickle
from PIL import Image
from collections import Counter
import tempfile
import av # Required for streamlit-webrtc video frame conversion
from tensorflow import keras # Import Keras from TensorFlow
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---------------------
# Page Configuration
# ---------------------
st.set_page_config(layout="wide", page_title="Bharatnatyam Mudra Recognizer")

# ---------------------
# Mudra Library Data
# ---------------------
mudra_library = {
    "Alapadmam": {"description": "A fully bloomed lotus.", "image": "mudra_images/Alapadmam.jpg"},
    "Anjali": {"description": "A gesture of reverence or salutation, where palms are joined.", "image": "mudra_images/Anjali.jpg"},
    "Aralam": {"description": "Bent or curved, like a finger holding a pot.", "image": "mudra_images/Aralam.jpg"},
    "Ardhachandran": {"description": "Represents the half-moon.", "image": "mudra_images/Ardhachandran.jpg"},
    "Ardhapathaka": {"description": "Represents a half-flag, a knife, or a tower.", "image": "mudra_images/Ardhapathaka.jpg"},
    "Berunda": {"description": "A mythical two-headed bird.", "image": "mudra_images/Berunda.jpg"},
    "Bramaram": {"description": "Represents a bee.", "image": "mudra_images/Bramaram.jpg"},
    "Chakra": {"description": "Represents a wheel or the discus of Lord Vishnu.", "image": "mudra_images/Chakra.jpg"},
    "Chandrakala": {"description": "Represents the digit or crescent of the moon.", "image": "mudra_images/Chandrakala.jpg"},
    "Chaturam": {"description": "Represents cleverness, eyes, or a square.", "image": "mudra_images/Chaturam.jpg"},
    "Garuda": {"description": "Represents the mythical eagle, mount of Lord Vishnu.", "image": "mudra_images/Garuda.jpg"},
    "Hamsapaksha": {"description": "Represents the wing of a swan.", "image": "mudra_images/Hamsapaksha.jpg"},
    "Hamsasyam": {"description": "Represents the beak of a swan.", "image": "mudra_images/Hamsasyam.jpg"},
    "Kangulam": {"description": "Represents a bud, a tail, or small bells.", "image": "mudra_images/Kangulam.jpg"},
    "Kapith": {"description": "Represents the wood apple or elephant apple.", "image": "mudra_images/Kapith.jpg"},
    "Kapotham": {"description": "Represents a dove or a pigeon.", "image": "mudra_images/Kapotham.jpg"},
    "Karkatta": {"description": "Represents a crab.", "image": "mudra_images/Karkatta.jpg"},
    "Kartariswastika": {"description": "Represents crossed scissors, branches of a tree, or a hill.", "image": "mudra_images/Kartariswastika.jpg"},
    "Katakamukha_1": {"description": "Represents opening a bracelet, plucking flowers, or holding a necklace.", "image": "mudra_images/Katakamukha_1.jpg"},
    "Katakamukha_2": {"description": "A variation of Katakamukha, used in different contexts.", "image": "mudra_images/Katakamukha_2.jpg"},
    "Katakamukha_3": {"description": "A variation of Katakamukha, used in different contexts.", "image": "mudra_images/Katakamukha_3.jpg"},
    "Katakavardhana": {"description": "Represents a couple, coronation, or worship.", "image": "mudra_images/Katakavardhana.jpg"},
    "Katrimukha": {"description": "Represents an arrow-notch or the face of scissors.", "image": "mudra_images/Katrimukha.jpg"},
    "Khatva": {"description": "Represents a bed or a cot.", "image": "mudra_images/Khatva.jpg"},
    "Kilaka": {"description": "Represents a bond of friendship or affection.", "image": "mudra_images/Kilaka.jpg"},
    "Kurma": {"description": "Represents a tortoise.", "image": "mudra_images/Kurma.jpg"},
    "Matsya": {"description": "Represents a fish.", "image": "mudra_images/Matsya.jpg"},
    "Mayura": {"description": "Represents a peacock.", "image": "mudra_images/Mayura.jpg"},
    "Mrigasirsha": {"description": "Represents the head of a deer.", "image": "mudra_images/Mrigasirsha.jpg"},
    "Mukulam": {"description": "Represents a flower bud or eating.", "image": "mudra_images/Mukulam.jpg"},
    "Mushti": {"description": "Represents a fist, grasping, or steadfastness.", "image": "mudra_images/Mushti.jpg"},
    "Nagabandha": {"description": "Represents entwined serpents.", "image": "mudra_images/Nagabandha.jpg"},
    "Padmakosha": {"description": "Represents a lotus bud or fruit.", "image": "mudra_images/Padmakosha.jpg"},
    "Pasha": {"description": "Represents a noose or a quarrel.", "image": "mudra_images/Pasha.jpg"},
    "Pathaka": {"description": "Represents a flag, the beginning of a dance, or blessings.", "image": "mudra_images/Pathaka.jpg"},
    "Pushpaputa": {"description": "Represents a handful of flowers for an offering.", "image": "mudra_images/Pushpaputa.jpg"},
    "Sakata": {"description": "Represents a cart or the wheel of a chariot.", "image": "mudra_images/Sakata.jpg"},
    "Samputa": {"description": "Represents a casket or a covered box.", "image": "mudra_images/Samputa.jpg"},
    "Sarpasirsha": {"description": "Represents the head of a serpent or a snake's hood.", "image": "mudra_images/Sarpasirsha.jpg"},
    "Shanka": {"description": "Represents a conch shell.", "image": "mudra_images/Shanka.jpg"},
    "Shivalinga": {"description": "Represents the Lingam, a symbol of Lord Shiva.", "image": "mudra_images/Shivalinga.jpg"},
    "Shukatundam": {"description": "Represents a parrot's beak.", "image": "mudra_images/Shukatundam.jpg"},
    "Sikharam": {"description": "Represents a peak, a spire, or the 'I' pronoun.", "image": "mudra_images/Sikharam.jpg"},
    "Simhamukham": {"description": "Represents the face of a lion.", "image": "mudra_images/Simhamukham.jpg"},
    "Suchi": {"description": "Represents a needle, one, or pointing.", "image": "mudra_images/Suchi.jpg"},
    "Swastikam": {"description": "Represents crossed hands, a symbol of prosperity or a welcome.", "image": "mudra_images/Swastikam.jpg"},
    "Tamarachudam": {"description": "Represents a rooster or a cock.", "image": "mudra_images/Tamarachudam.jpg"},
    "Tripathaka": {"description": "Represents the three parts of a flag, a crown, or a tree.", "image": "mudra_images/Tripathaka.jpg"},
    "Trishulam": {"description": "Represents a trident, the weapon of Lord Shiva.", "image": "mudra_images/Trishulam.jpg"},
    "Varaha": {"description": "Represents a boar.", "image": "mudra_images/Varaha.jpg"},
}

# ---------------------
# Model Loading
# ---------------------
MODEL_DIR = "model_artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Use st.cache_resource to load model and artifacts only once
@st.cache_resource
def load_artifacts():
    """Loads the trained model, scaler, and label encoder."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please run model_train.py first to generate the artifacts.")
        return None, None, None
    
    model = keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)
    return model, scaler, le

model, scaler, le = load_artifacts()
if model is None:
    st.stop()

# ---------------------
# MediaPipe & Prediction Utilities
# ---------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image_bgr):
    """
    Extracts hand landmarks from a BGR image.
    Returns the landmark vector (63,) and the landmarks object, or (None, None).
    """
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
            return coords, landmarks
    return None, None

def predict_from_landmarks(landmark_vector):
    """
    Takes a landmark vector and returns the predicted mudra label and confidence.
    """
    # Reshape and scale the input vector
    X_scaled = scaler.transform(landmark_vector.reshape(1, -1))
    
    # Make prediction
    probabilities = model.predict(X_scaled, verbose=0)[0]
    prediction_index = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction_index])
    
    # Decode the label
    label = le.inverse_transform([prediction_index])[0]
    return label, confidence

# ---------------------
# Streamlit UI
# ---------------------
st.title("Bharatnatyam Mudra Recognizer")
st.markdown("An interactive tool to identify Bharatnatyam hand gestures (mudras) from images, videos, or your live webcam feed.")

mode = st.sidebar.radio(
    "Choose your mode",
    ["Mudra Library", "Image", "Webcam Snapshot", "Video", "Live Webcam Feed"]
)

# --- Mode: Mudra Library ---
if mode == "Mudra Library":
    st.header("Mudra Library")
    st.write("Browse through the different mudras and their meanings.")
    
    cols = st.columns(4)
    sorted_mudras = sorted(mudra_library.items())
    
    for i, (name, data) in enumerate(sorted_mudras):
        with cols[i % 4]:
            st.subheader(name)
            img_path = data.get("image")
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_column_width='always')
            else:
                st.warning(f"Image for {name} not found.")
            st.caption(data.get("description", "No description available."))
            st.markdown("---")

# --- Mode: Image Upload ---
elif mode == "Image":
    st.header("Identify Mudra from an Image")
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        landmark_vector, landmarks_obj = extract_landmarks(bgr_image)
        
        if landmark_vector is None:
            st.warning("No hand detected in the image. Please try another one.")
        else:
            label, confidence = predict_from_landmarks(landmark_vector)
            
            with col2:
                st.success(f"**Predicted Mudra:** {label}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
                # Display info from library
                entry = mudra_library.get(label)
                if entry:
                    st.write(f"**Description:** {entry['description']}")
                    if os.path.exists(entry['image']):
                        st.image(entry['image'], caption="Reference Image", width=200)
                
                # Draw landmarks on the image and display
                img_with_landmarks = bgr_image.copy()
                mp_drawing.draw_landmarks(img_with_landmarks, landmarks_obj, mp_hands.HAND_CONNECTIONS)
                st.image(cv2.cvtColor(img_with_landmarks, cv2.COLOR_BGR2RGB), caption="Detected Hand Landmarks", use_column_width=True)

# --- Mode: Webcam Snapshot ---
elif mode == "Webcam Snapshot":
    st.header("Identify Mudra from a Webcam Snapshot")
    img_file_buffer = st.camera_input("Position your hand and take a photo")

    if img_file_buffer:
        image = Image.open(img_file_buffer).convert("RGB")
        bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        landmark_vector, landmarks_obj = extract_landmarks(bgr_image)
        
        if landmark_vector is None:
            st.warning("No hand detected in the snapshot. Please try again.")
        else:
            label, confidence = predict_from_landmarks(landmark_vector)
            st.success(f"**Predicted Mudra:** {label} ({confidence:.2%})")

            # Display info from library
            entry = mudra_library.get(label)
            if entry:
                col1, col2 = st.columns([1, 2])
                with col1:
                    if os.path.exists(entry['image']):
                        st.image(entry['image'], use_column_width=True)
                with col2:
                    st.write(f"**{label}**")
                    st.write(entry['description'])

# --- Mode: Video Upload ---
elif mode == "Video":
    st.header("Identify Mudras from a Video")
    uploaded_video = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
    sample_rate = st.slider("Process one frame every 'N' frames", 1, 30, 10, help="Higher values process faster but may miss quick gestures.")

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.info(f"Video has {frame_count} frames. Processing will take a moment...")
        
        results = []
        p_bar = st.progress(0, text="Analyzing video...")
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                lm_vec, _ = extract_landmarks(frame)
                if lm_vec is not None:
                    label, conf = predict_from_landmarks(lm_vec)
                    results.append((label, conf))
            
            p_bar.progress(frame_idx / frame_count, text=f"Analyzing frame {frame_idx}/{frame_count}")
            frame_idx += 1
            
        cap.release()
        os.remove(video_path)
        p_bar.progress(1.0, text="Analysis Complete!")

        if not results:
            st.warning("No hands were detected in the processed video frames.")
        else:
            st.subheader("Detected Mudras Summary")
            labels = [r[0] for r in results]
            counts = Counter(labels)
            
            for label, count in counts.most_common():
                entry = mudra_library.get(label)
                if entry:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if os.path.exists(entry['image']):
                            st.image(entry['image'], width=150)
                    with col2:
                        st.write(f"**{label}** (Detected {count} times)")
                        st.write(entry['description'])
                    st.markdown("---")

# --- Mode: Live Webcam Feed ---
elif mode == "Live Webcam Feed":
    st.header("Live Mudra Detection")
    st.write("Allow camera access. The model will identify the mudra you are showing in real-time.")

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class MudraVideoProcessor(VideoProcessorBase):
        def __init__(self):
            # We initialize a new Hands object for each processor instance.
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            # Process the image to find hand landmarks
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            label_text = ""
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark vector for prediction
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
                
                try:
                    pred_label, pred_prob = predict_from_landmarks(coords)
                    label_text = f"{pred_label} ({pred_prob:.2f})"
                except Exception as e:
                    label_text = "Prediction Error"
            
            # Put the predicted label on the image
            cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live-mudra-detection",
        video_processor_factory=MudraVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )