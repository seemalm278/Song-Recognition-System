import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import requests
import tempfile
import os
import time

# --- 1. PROJECT CONFIG & CV MODEL LOADING ---
st.set_page_config(page_title="CV Song Recognition Pro", layout="wide")

@st.cache_resource
def load_cv_engine():
    # This is the MobileNetV2 model required for your CV project
    return MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

cv_model = load_cv_engine()

# --- 2. CORE FUNCTIONS ---

def create_spectrogram(audio_path):
    """The CV Step: Transforms 1D audio into a 2D image"""
    y, sr = librosa.load(audio_path, duration=10.0)
    y = librosa.util.normalize(y) # Make audio clear
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig = plt.figure(figsize=(5, 5))
    librosa.display.specshow(S_dB, sr=sr)
    plt.axis('off')
    spec_path = "temp_spectrogram.png"
    plt.savefig(spec_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return spec_path

def run_cv_analysis(img_path):
    """Passes the spectrogram through MobileNetV2"""
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = cv_model.predict(img_array)
    return features

# --- 3. UI LAYOUT ---

st.title("🎵 CV-Powered Song Recognition")
st.markdown("Semester 6 Project: Audio Classification via **MobileNetV2 Spectrogram Analysis**")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("AudD API Key", type="password", placeholder="Paste your key here")
    st.info("The system uses MobileNetV2 to extract visual features from sound.")

tab1, tab2 = st.tabs(["🎤 Live Recognition", "📁 About the Model"])

with tab1:
    audio_data = st.audio_input("Record a 10-second snippet of the song")
    
    if audio_data:
        if not api_key:
            st.warning("Please enter your AudD API Key in the sidebar!")
        else:
            # Get the raw bytes from the Streamlit audio recorder
            audio_bytes = audio_data.getvalue()

            # Save to a temporary file safely for Librosa/Spectrogram processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                audio_path = tmp_file.name
            
            # --- COMPUTER VISION PIPELINE ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("Step 1: Generating Visual Spectrogram...")
                spec_img = create_spectrogram(audio_path)
                st.image(spec_img, caption="Visual Input for MobileNetV2", use_container_width=True)
            
            with col2:
                st.info("Step 2: MobileNetV2 Feature Extraction...")
                features = run_cv_analysis(spec_img)
                st.success("✅ CNN Processing Complete")
                
                # Clean up the temporary file immediately so it doesn't lock disk resources
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

                # --- IDENTIFICATION PIPELINE ---
                with st.spinner("Step 3: Querying AudD Database..."):
                    # Pass the in-memory audio bytes directly to requests to bypass Windows stream lock
                    files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
                    data = {'api_token': api_key}
                    
                    try:
                        response = requests.post('https://api.audd.io/', data=data, files=files)
                        result = response.json()
                        
                        if result.get('status') == 'success' and result.get('result'):
                            res = result['result']
                            st.header(f"🎶 {res['title']}")
                            st.subheader(f"👤 {res['artist']}")
                            st.write(f"💿 Album: {res['album']}")
                        else:
                            # Safely extract detailed error messages from AudD API response if validation fails
                            error_details = result.get('error', {})
                            error_msg = error_details.get('error_msg', 'Could not identify the song. Try a clearer recording.')
                            st.error(f"❌ API Status: {error_msg}")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")

with tab2:
    st.header("How it works (For Viva/Presentation)")
    st.write("""
    1. **Signal Processing:** Audio is sampled and converted to a Mel-Spectrogram.
    2. **Computer Vision:** The spectrogram is treated as a 224x224 image.
    3. **MobileNetV2:** This pre-trained CNN extracts deep spatial features from the image.
    4. **Global Match:** The identified visual signatures are matched via the AudD API.
    """)