"""
Pre-trained Audio Model for Music Genre Classification
Uses state-of-the-art pre-trained models for better accuracy
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PretrainedAudioClassifier:
    """Uses pre-trained audio models for music genre classification."""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_yamnet_model(self):
        """Load YAMNet pre-trained model."""
        try:
            # Load YAMNet from TensorFlow Hub
            yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            logger.info("YAMNet model loaded successfully")
            return yamnet_model
        except Exception as e:
            logger.error(f"Failed to load YAMNet: {e}")
            return None
    
    def extract_yamnet_features(self, audio_path: str, yamnet_model) -> np.ndarray:
        """Extract features using YAMNet."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)  # YAMNet expects 16kHz
            
            # Get YAMNet embeddings
            scores, embeddings, spectrogram = yamnet_model(audio)
            
            # Use the embeddings as features (average across time)
            features = np.mean(embeddings.numpy(), axis=0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting YAMNet features from {audio_path}: {e}")
            return None
    
    def build_classifier_model(self, input_dim: int) -> keras.Model:
        """Build classifier on top of pre-trained features."""
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,))
        
        # Dense layers for classification
        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_features_from_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features from all audio files using YAMNet."""
        
        # Load YAMNet model
        yamnet_model = self.load_yamnet_model()
        if yamnet_model is None:
            logger.error("Could not load YAMNet model")
            return None, None, None
        
        features_list = []
        labels_list = []
        genre_names = []
        
        genre_dirs = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))]
        
        logger.info(f"Processing {len(genre_dirs)} genres with YAMNet...")
        
        for genre in genre_dirs:
            genre_path = os.path.join(data_dir, genre)
            audio_files = [f for f in os.listdir(genre_path) 
                          if f.endswith(('.mp3', '.wav', '.m4a'))]
            
            logger.info(f"Processing {len(audio_files)} files for genre: {genre}")
            
            for audio_file in audio_files:
                audio_path = os.path.join(genre_path, audio_file)
                features = self.extract_yamnet_features(audio_path, yamnet_model)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(genre)
                    genre_names.append(genre)
        
        if not features_list:
            logger.error("No features extracted!")
            return None, None, None
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        logger.info(f"Extracted {len(X)} feature vectors with {X.shape[1]} features each")
        
        return X, y, genre_names
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> keras.callbacks.History:
        """Train the classifier."""
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_onehot = tf.keras.utils.to_categorical(y_encoded)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_onehot, test_size=0.2, stratify=y_onehot, random_state=42
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        # Build and train model
        self.model = self.build_classifier_model(X.shape[1])
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_pretrained_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate the model."""
        # Make predictions
        y_pred_proba = self.model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        # Print classification report
        print("\n" + "="*60)
        print("PRETRAINED MODEL CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))
        
        

def main():
    """Main function to run pre-trained audio classification."""
    data_dir = "Data"
    
    # Initialize classifier
    classifier = PretrainedAudioClassifier(num_classes=10)
    
    # Extract features using YAMNet
    logger.info("Extracting features using YAMNet...")
    X, y, genre_names = classifier.extract_features_from_dataset(data_dir)
    
    if X is None:
        logger.error("Failed to extract features")
        return
    
    # Train model
    logger.info("Training pre-trained model...")
    history = classifier.train(X, y, epochs=100)
    
    # Evaluate
    y_encoded = classifier.label_encoder.transform(y)
    y_onehot = tf.keras.utils.to_categorical(y_encoded)
    _, X_val, _, y_val = train_test_split(
        X, y_onehot, test_size=0.2, stratify=y_onehot, random_state=42
    )
    
    classifier.evaluate(X_val, y_val)
    
    logger.info("Pre-trained model training complete!")

if __name__ == "__main__":
    main() 