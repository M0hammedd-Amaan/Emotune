import os
import streamlit as st
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Function to preprocess audio
def preprocess_audio(file_path, max_frames=236):
    try:
        value, sample = librosa.load(file_path)
        noise_amp = 0.035 * np.random.uniform() * np.amax(value)
        value = value + noise_amp * np.random.normal(size=value.shape[0])
        mfcc = librosa.feature.mfcc(y=value, sr=sample, n_mfcc=20)
        mfcc = np.expand_dims(mfcc, axis=-1)
        chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(value)), sr=sample).T, axis=0)
        chroma = np.expand_dims(chroma, axis=0)
        chroma = np.repeat(chroma, mfcc.shape[0], axis=0)
        chroma = np.expand_dims(chroma, axis=-1)
        result = np.concatenate((mfcc, chroma), axis=1)
        
        if result.shape[1] < max_frames:
            padding = np.zeros((result.shape[0], max_frames - result.shape[1], 1))
            result = np.concatenate((result, padding), axis=1)
        elif result.shape[1] > max_frames:
            result = result[:, :max_frames, :]
            
        result = result.reshape(-1)
        return result
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return None

# Function to predict emotion from a preprocessed audio file
def predict_emotion(file_path, model, scaler, max_frames=236):
    features = preprocess_audio(file_path, max_frames=max_frames)
    if features is not None:
        features = features.reshape(1, -1)  # Flatten to 1D array
        features = scaler.transform(features)  # Standardize
        features = features.reshape(features.shape[0], features.shape[1], 1)  # Reshape for Conv1D
        prediction = model.predict(features)
        return np.argmax(prediction[0])
    else:
        return None

# Update the path to your model and scaler file here
model_path = r'C:\Users\amaan\Desktop\TESS Toronto emotional speech set data\emotion_recognition_model.h5'
scaler_path = r'C:\Users\amaan\Desktop\TESS Toronto emotional speech set data\standard_scaler.pkl'

# Load model
model = load_model(model_path)

# Load scaler from file
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error(f"Scaler file not found at: {scaler_path}")
    st.stop()

# Mapping emotions to Spotify playlist URLs
emotion_to_url = {
    "happy": "https://open.spotify.com/playlist/4nd7oGDNgfM0rv28CQw9WQ?si=7dad1492c8ef4bc1",
    "sad": "https://open.spotify.com/playlist/6vEYfoIeIIzC87k94JDcYj?si=e7569fd4ad7b4467",
    "angry": "https://open.spotify.com/playlist/3p8ejB7BscAVmEdyK7AtXx?si=7bd5292b212e47a3",
    "disgust": "https://open.spotify.com/playlist/0CiM52P324UEuHEX7cYSi8?si=3e277c196f574df9",
    "neutral": "https://open.spotify.com/playlist/7EClwmhqu7mg4JvUI9z5DT?si=ee6d871b0c654984",
    "fear": "https://open.spotify.com/playlist/2zzLBcgMuP5SYZGEjO48E3?si=690f0a92d723412b",
    "unknown": "https://open.spotify.com/playlist/1tPFKAMSJgckqRgaqP8wFt?si=a945b7492a894f70"
}

# Function to get URL based on emotion
def get_url_for_emotion(emotion):
    return emotion_to_url.get(emotion, "https://open.spotify.com/playlist/1tPFKAMSJgckqRgaqP8wFt?si=a945b7492a894f70")

# Streamlit app
st.title("EMOTUNE")
st.write("Upload an audio file to detect the emotion and get a corresponding Spotify playlist.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    temp_file_path = "temp.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format='audio/wav')

    predicted_emotion = predict_emotion(temp_file_path, model, scaler)
    if predicted_emotion is not None:
        predicted_emotion_label = ["angry", "disgust", "fear", "happy", "neutral", "sad", "unknown"][predicted_emotion]
        st.write(f"Detected emotion: {predicted_emotion_label}")
        playlist_url = get_url_for_emotion(predicted_emotion_label)
        st.markdown(f'<center><a href="{playlist_url}" target="_blank" style="padding: 10px; background-color: #1DB954; color: white; text-decoration: none; border-radius: 5px;">Open Spotify Playlist</a></center>', unsafe_allow_html=True)
    else:
        st.write("Could not detect emotion.")
else:
    st.write("Please upload a file to proceed.")
