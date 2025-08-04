import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import librosa
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
app.secret_key = 'musaic_secret_key_2024'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class MUSAICClassifier:
    def __init__(self, model_path: str = "MUSAIC.h5"):
        self.model_path = model_path
        self.model = None
        self.yamnet_model = None
        self.label_encoder = None
        self.load_model()
    
    def load_model(self):
        try:
            self.model = keras.models.load_model(self.model_path)
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            self.label_encoder = ['blues', 'classical', 'country', 'disco', 
                                'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            scores, embeddings, spectrogram = self.yamnet_model(audio)
            features = np.mean(embeddings.numpy(), axis=0)
            return features
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def predict_genre(self, audio_path: str) -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        features = self.extract_features(audio_path)
        if features is None:
            raise ValueError("Failed to extract features from audio file")
        
        features = features.reshape(1, -1)
        predictions = self.model.predict(features, verbose=0)
        
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_genre = self.label_encoder[predicted_class]
        
        genre_probs = {}
        for i, genre in enumerate(self.label_encoder):
            genre_probs[genre] = float(predictions[0][i])
        
        return {
            'predicted_genre': predicted_genre,
            'confidence': confidence,
            'genre_probabilities': genre_probs,
            'file_path': audio_path
        }

classifier = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = classifier.predict_genre(filepath)
            return render_template('result.html', result=result, filename=filename)
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload MP3, WAV, M4A, or FLAC files.')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            result = classifier.predict_genre(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': classifier is not None})

if __name__ == '__main__':
    try:
        print("Loading MUSAIC model...")
        classifier = MUSAICClassifier()
        print("Model loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Make sure MUSAIC.h5 is in the current directory") 