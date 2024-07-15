import os
import pandas as pd
import numpy as np
import joblib
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import accuracy_score

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
        print(f"Error processing {file_path}: {e}")
        return None

# Paths and dataset setup
drive_path = r'C:\Users\amaan\Desktop\emotune\ml'
os.chdir(os.path.join(drive_path, 'TESS Toronto emotional speech set data'))
path = os.path.join(drive_path, 'TESS Toronto emotional speech set data')
audio_path = []
audio_emotion = []

directory_path = os.listdir(path)
for audio in directory_path:
    if not audio.endswith('.wav'):
        continue
    audio_path.append(os.path.join(path, audio))
    emotion = audio.split('_')[-1].split('.')[0]
    if emotion == 'sad':
        audio_emotion.append("sad")
    elif emotion == 'angry':
        audio_emotion.append("angry")
    elif emotion == 'disgust':
        audio_emotion.append("disgust")
    elif emotion == 'neutral':
        audio_emotion.append("neutral")
    elif emotion == 'happy':
        audio_emotion.append("happy")
    elif emotion == 'fear':
        audio_emotion.append("fear")
    else:
        audio_emotion.append("unknown")

emotion_dataset = pd.DataFrame(audio_emotion, columns=['Emotions'])
audio_path_dataset = pd.DataFrame(audio_path, columns=['Path'])
dataset = pd.concat([audio_path_dataset, emotion_dataset], axis=1)

X, Y = [], []
print("Feature processing...")
max_frames = 236
for path, emo in zip(dataset.Path, dataset.Emotions):
    features = preprocess_audio(path, max_frames=max_frames)
    if features is not None:
        X.append(features)
        Y.append(emo)

print("Feature processing completed.")

X = np.array(X)
Y = np.array(Y)

# Verify the unique classes in your dataset
unique_classes = np.unique(Y)
print(f"Unique classes in the dataset: {unique_classes}")
print(f"Number of unique classes: {len(unique_classes)}")

# Save the extracted features and labels to a CSV file
df = pd.DataFrame(X)
df['Emotions'] = Y
df.to_csv('data.csv', index=False)

# Load data and labels from CSV file
data = pd.read_csv('data.csv')
print(data.columns)  # Print columns to debug

X = data.drop('Emotions', axis=1).values
y = data['Emotions'].values

# Reshape X to 3D for Conv1D layer
X = X.reshape(X.shape[0], X.shape[1], 1)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[1])).reshape(X.shape)
scaler_path = r'C:\Users\amaan\Desktop\TESS Toronto emotional speech set data\standard_scaler.pkl'
joblib.dump(scaler, scaler_path)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding, ensuring all classes are included
all_classes = list(unique_classes)
y_train = pd.get_dummies(y_train).reindex(columns=all_classes, fill_value=0).values.astype('float32')
y_test = pd.get_dummies(y_test).reindex(columns=all_classes, fill_value=0).values.astype('float32')

# CNN model
model = Sequential()
model.add(Conv1D(256, 5, padding='same', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 5, padding='same'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(unique_classes), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model
model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('emotion_recognition_model.h5')

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Model: CNN    Accuracy: {accuracy * 100:.2f}%")

# Mapping emotions to URLs
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

# Example usage
audio_file_path = r"C:\Users\amaan\Downloads\test.wav"
predicted_emotion = predict_emotion(audio_file_path, model, scaler)

if predicted_emotion is not None:
    predicted_emotion_label = unique_classes[predicted_emotion]
    url = get_url_for_emotion(predicted_emotion_label)
    print(f"Detected emotion: {predicted_emotion_label}")
    print(f"Playlist URL: {url}")
else:
    print("Could not detect emotion.")
