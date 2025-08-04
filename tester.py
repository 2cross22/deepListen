import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import librosa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractivePredictor:
    
    def __init__(self, model_path: str = "MUSAIC.h5"):
        self.model_path = model_path
        self.model = None
        self.yamnet_model = None
        self.label_encoder = None
        
        print("Loading Music Genre Classifier...")
        self.load_model()
        print("Ready to predict genres!")
    
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
        
        print(f"Analyzing: {os.path.basename(audio_path)}")
        
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
    
    def print_prediction_result(self, result: dict):
        print("\n" + "="*60)
        print(f"SONG: {os.path.basename(result['file_path'])}")
        print("="*60)
        
        print(f"PREDICTED GENRE: {result['predicted_genre'].upper()}")
        print(f"CONFIDENCE: {result['confidence']:.2%}")
        
        print("\nALL GENRE PROBABILITIES:")
        print("-" * 40)
        
        sorted_probs = sorted(result['genre_probabilities'].items(), 
                             key=lambda x: x[1], reverse=True)
        
        for genre, prob in sorted_probs:
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"{genre:10} {prob:.2%} {bar}")
    
    def run_interactive(self):
        print("\nMUSIC GENRE CLASSIFIER (90% Accuracy)")
        print("="*50)
        print("Supported formats: MP3, WAV, M4A")
        print("Type 'quit' to exit")
        print("="*50)
        
        while True:
            try:
                file_path = input("\nEnter path to your audio file: ").strip()
                
                if file_path.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not file_path:
                    print("Please enter a file path")
                    continue
                
                result = self.predict_genre(file_path)
                self.print_prediction_result(result)
                
                another = input("\nTry another song? (y/n): ").strip().lower()
                if another not in ['y', 'yes', '']:
                    print("Thanks for using the Music Genre Classifier!")
                    break
                
            except FileNotFoundError as e:
                print(f"File not found: {e}")
                print("Make sure the file path is correct")
                
            except Exception as e:
                print(f"Error: {e}")
                print("Make sure the file is a valid audio file")

def main():
    try:
        model_path = "MUSAIC.h5"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("Make sure you've trained the model first")
            return
        
        predictor = InteractivePredictor(model_path)
        predictor.run_interactive()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure:The model file 'MUSAIC.h5' is in the current directory")

if __name__ == "__main__":
    main() 